"""V1a: classifier2's loss (expectation form) and the ideal-image win-rate evaluation on the frozen SimCLR encoder (head only).

New code in this file: the V1a loss / fit (``v1a_loss``, ``fit_v1a``), the trajectory-image set T (``trajectory_set``), the win-rate prediction with macro-F1 (``win_rate_predict``) and the context
subclass that routes the repository's sequential judgment-unit protocol through them.  Everything else is the repository's code (see REPORT_V1a.md for file:line):

  data / ideal split / splits ........ judgment_unit_study.Context (-> run_pair_endpoint_study.setup, pair_preference_endpoint.split_classifier2_style)
  acquisition rules .................. judgment_unit_strategies.make_selector / ENSEMBLE (Random: judgment_unit_study.random_picks)
  sequential protocol ................ judgment_unit_study.trajectory (rounds of 10 judgments, retrain after every round)
  secondary endpoint ................. pair_preference_endpoint.evaluate_full
  primary endpoint ................... frozen_encoder_reward_head.evaluate_model (accuracy) — checked against ``win_rate_predict`` in test_v1a.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY)); sys.path.insert(0, str(STUDY.parents[1] / "active_learning_program"))

import frozen_encoder_reward_head as frozen  # noqa: E402
import judgment_unit_study as ju_study  # noqa: E402
import judgment_unit_strategies as ju  # noqa: E402
import pair_preference_endpoint as endpoint  # noqa: E402
import run_pair_endpoint_study as single_study  # noqa: E402

DATA = single_study.DATA
CACHE_FILES = (single_study.CACHE, STUDY.parent / "new_directions" / "results" / "feature_cache_all.pt")
ACTIVE = ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")          # the four classified types (Twinned, index 1, is excluded from the study)
COL = frozen.TYPE_TO_INDEX                                       # type name -> output column
MIX = (0.60, 0.25, 0.15)                                         # classifier2: pairwise / ideal anchoring / bad image
IDEAL_VS_TRAJECTORY = 0.7                                        # classifier2: 70 % ideal vs trajectory, 30 % ideal vs other-type ideal
WEIGHT_DECAY = 1e-4; CLIP = 1.0
SEEDS = (42, 79, 123, 202, 303) + tuple(range(400, 430))        # identical to V2 (35 seeds)
RULES = ("random", "uncertainty", "core_set", "typiclust_pairs", "laplace_bald", "bald_decisive", "ensemble_bald", "probcover_pairs")
RANDOM_REPLICATES = ju.RANDOM_REPLICATES                         # Random = mean over 5 draws (repository convention of V2)


def load_cache(paths=CACHE_FILES) -> dict:
    cache: dict = {}
    for p in paths: cache.update(frozen.load_feature_cache(p, DATA))
    return cache


# ------------------------------------------------------------------------------------------------ trajectory images T
_HASHES: dict = {}


def sha256_of(path) -> str:
    key = str(path)
    if key not in _HASHES: _HASHES[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _HASHES[key]


def all_ideal_files() -> list[str]:
    """Every ideal image file of the five ideal directories (classifier2 ``IDEAL_DIRS``), including Twinned."""
    root = DATA / "original data"
    return sorted(str(p.resolve()) for d in sorted(root.glob("STO_ideal_*")) for p in d.iterdir() if p.suffix.lower() in (".png", ".bmp"))


def trajectory_set(cache: dict) -> tuple[list[str], dict]:
    """T = all trajectory images of the feature cache minus every image whose bytes (SHA-256) equal those of any ideal image."""
    files = all_ideal_files(); ideal = {sha256_of(p) for p in files}; traj = sorted(p for p in cache if "Trajectories" in p)
    keep = [p for p in traj if sha256_of(p) not in ideal]
    return keep, {"n_trajectory_images": len(traj), "n_removed_identical_to_ideal": len(traj) - len(keep), "n_ideal_files": len(files), "n_unique_ideal_contents": len(ideal), "T": len(keep)}


# ------------------------------------------------------------------------------------------------ loss
def pair_loss(a: torch.Tensor, b: torch.Tensor, winner: np.ndarray, weight: torch.Tensor) -> torch.Tensor:
    """(1/n) sum_j c_j l_j with l = -log sigmoid(d) (y=1), -log sigmoid(-d) (y=2), |d| (tie), relu(a)+relu(b) (not_apply); d = a - b (own-type scores)."""
    m = {l: torch.as_tensor(winner == l) for l in frozen.LABELS}
    terms = -F.logsigmoid(a - b) * m["1"] - F.logsigmoid(b - a) * m["2"] + (a - b).abs() * m["tie"] + (F.relu(a) + F.relu(b)) * m["not_apply"]
    return (terms * weight).sum() / max(len(a), 1)


def ideal_loss(sR: dict, sT: torch.Tensor) -> torch.Tensor:
    """Mean over the four types t of 0.7 * mean_{x in R_t, u in T} -log sig(s_t(x)-s_t(u)) + 0.3 * mean_{o != t} mean_{x in R_t, x' in R_o} -log sig(s_t(x)-s_t(x')); ``sR[name]`` = scores of the reference images of that type."""
    per_type = []
    for t in sR:
        c = COL[t]; x = sR[t][:, c]
        vs_traj = F.softplus(sT[:, c][None, :] - x[:, None]).mean()
        vs_other = torch.stack([F.softplus(sR[o][:, c][None, :] - x[:, None]).mean() for o in sR if o != t]).mean()
        per_type.append(IDEAL_VS_TRAJECTORY * vs_traj + (1 - IDEAL_VS_TRAJECTORY) * vs_other)
    return torch.stack(per_type).mean()


def bad_loss(sB: torch.Tensor, sT: torch.Tensor) -> torch.Tensor:
    """Mean over the five outputs tau of mean_{bad v, trajectory u} -log sigmoid(s_tau(u) - s_tau(v))."""
    return F.softplus(sB[:, None, :] - sT[None, :, :]).mean()


def v1a_loss(head, xa, xb, typ, weight, winner, refs: dict, traj: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    index = torch.arange(len(typ)); a, b = head(xa)[index, typ], head(xb)[index, typ]
    sR = {t: head(f) for t, f in refs.items()}; sT = head(traj); sB = head(bad)
    return MIX[0] * pair_loss(a, b, winner, weight) + MIX[1] * ideal_loss(sR, sT) + MIX[2] * bad_loss(sB, sT)


def fit_v1a(exp, features, rows, traj: torch.Tensor, lr: float, steps: int, head_seed: int | None = None):
    """Full-batch AdamW (weight decay 1e-4, gradient-norm clip 1.0) on the V1a loss.  Returns a ``BTModel`` copy (the repository's model class) with the trained head.
    The initial head is the experiment seed's own initialisation (identical at every retrain) unless ``head_seed`` re-initialises it (ensemble members), as in ``frozen.train_model``."""
    base = features.base_model; model = copy.copy(base); model._modules = copy.copy(base._modules)   # the frozen encoder is shared (never modified), only the head is copied
    model.reward_head = copy.deepcopy(base.reward_head)
    if head_seed is not None:
        torch.manual_seed(head_seed)
        for layer in model.reward_head:
            if hasattr(layer, "reset_parameters"): layer.reset_parameters()
    head = model.reward_head.eval()                                      # dropout inert during training, as in frozen.fit_head (full-batch, order independent)
    refs = {t: features.get(exp.references[t]) for t in ACTIVE}; bad = features.get(exp.bad_paths)
    xa, xb = features.get(rows.resolved_img1), features.get(rows.resolved_img2); typ = torch.as_tensor(rows.type_idx.to_numpy())
    weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    for _ in range(steps):
        loss = v1a_loss(head, xa, xb, typ, weight, winner, refs, traj, bad)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP); opt.step()
    return model.eval()


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def win_rate_predict(exp, features, model, split: str = "outer_test") -> tuple[list[str], list[str]]:
    """(truth, prediction) for every ideal image of ``split`` ('outer_test' = the 28 test ideal images, 'utility' = the 28 validation ideal images):
    W_c(x) = mean over reference images z of every other type of sigmoid(s_c(x) - s_c(z)); prediction = argmax_c W_c(x) over the four types (first maximum on ties)."""
    images = exp.test_images if split == "outer_test" else exp.utility_images; head = model.reward_head.eval()
    refs = {c: head(features.get(ps)).numpy() for c, ps in exp.references.items()}; truth, pred = [], []
    for label, paths in images.items():
        for score in head(features.get(paths)).numpy():
            win = {}
            for c in refs:
                opp = np.concatenate([s for o, s in refs.items() if o != c]); win[c] = float(np.mean(1 / (1 + np.exp(-np.clip(score[COL[c]] - opp[:, COL[c]], -50, 50)))))
            truth.append(label); pred.append(max(win, key=win.get))
    return truth, pred


def accuracy_and_macro_f1(truth, pred) -> tuple[float, float]:
    acc = float(np.mean([t == p for t, p in zip(truth, pred)])); f1s = []
    for c in ACTIVE:
        tp = sum(t == c and p == c for t, p in zip(truth, pred)); fp = sum(t != c and p == c for t, p in zip(truth, pred)); fn = sum(t == c and p != c for t, p in zip(truth, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return acc, float(np.mean(f1s))


# ------------------------------------------------------------------------------------------------ context
class V1aContext(ju_study.Context):
    """The repository's judgment-unit context (split B = classifier2-style 20 % pair hold-out) with the V1a fit and the V1a evaluation."""
    def __init__(self, seed, scratch, cache, lr, steps):
        super().__init__(seed, Path(scratch), cache, "B", "groups", lr, steps, None)
        keep, self.t_info = trajectory_set(cache); assert all(p in cache for p in keep); self.traj = self.features.get(keep); self._fits: dict = {}
        assert not self.validation, "split B has no validation pair groups"

    def fit(self, judgments, head_seed=None, lr=None, steps=None):
        lr0, steps0 = self.params_for(judgments); lr = lr0 if lr is None else lr; steps = steps0 if steps is None else steps
        key = (tuple(sorted(set(judgments))), head_seed, lr, steps)
        if key not in self._fits: self._fits[key] = fit_v1a(self.exp, self.features, ju_study.rows_of(self.exp, judgments), self.traj, lr, steps, head_seed)
        return self._fits[key]

    def evaluate(self, model) -> dict:
        truth, pred = win_rate_predict(self.exp, self.features, model, "outer_test"); acc, f1 = accuracy_and_macro_f1(truth, pred)
        vt, vp = win_rate_predict(self.exp, self.features, model, "utility"); vacc, _ = accuracy_and_macro_f1(vt, vp)
        sec = endpoint.evaluate_full(self.exp, self.features, model, self.validation, self.test)
        return {"test_ideal_accuracy": acc, "test_ideal_macro_f1": f1, "val_ideal_accuracy": vacc, "n_test_ideal": len(truth), "pref_auc": sec["test_decisive_auc"],
                "pref_accuracy": sec["test_decisive_accuracy"], "pref_log_loss": sec["test_decisive_log_loss"], "pref_rows": sec["test_decisive_rows"]}

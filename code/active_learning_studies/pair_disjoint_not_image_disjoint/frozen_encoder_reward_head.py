"""Reward-model training on a frozen, cached encoder.

The SimCLR encoder is applied once to every image and never updated.  Only the reward head is trained,
by full-batch gradient descent on a *sum* of per-row losses.  A sum does not depend on row order, and
the head has no dropout while training, so the trained head is a function of the **set** of labelled
pair groups: reordering the same pair groups cannot change the result (up to floating-point addition
order).  Dropout stays in the head so MC-dropout acquisition can still sample it at selection time.

Losses mirror ``Experiment.train``: Bradley--Terry for decisive winners, an absolute-difference term for
ties, a push-down term for ``not_apply``, reference-anchor ranking, and bad-image negative anchors.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "active_learning_program"))
import pairwise_active_learning_pipeline as pipeline  # noqa: E402
from pairwise_active_learning_pipeline import BTModel, Experiment, TYPE_TO_INDEX, transform  # noqa: E402

LABELS = ("1", "2", "tie", "not_apply")


class FrozenFeatures:
    """Encodes each image path once with the untrained-by-us SimCLR encoder and caches the 512-d result."""
    def __init__(self, exp: Experiment, cache: dict[str, torch.Tensor] | None = None):
        self.base_model: BTModel = exp.make_model().eval()
        for parameter in self.base_model.encoder.parameters(): parameter.requires_grad_(False)
        self.tf = transform(); self.cache: dict[str, torch.Tensor] = {} if cache is None else cache  # may be shared across seeds: same encoder

    @torch.no_grad()
    def get(self, paths: Iterable) -> torch.Tensor:
        keys = [str(p) for p in paths]; missing = [k for k in dict.fromkeys(keys) if k not in self.cache]
        for i in range(0, len(missing), 16):
            chunk = missing[i:i + 16]
            batch = torch.stack([self.tf(Image.open(k).convert("L")) for k in chunk])
            for k, v in zip(chunk, self.base_model.encoder(batch)): self.cache[k] = v
        return torch.stack([self.cache[k] for k in keys]) if keys else torch.empty(0, 512)

    def embedding_cache(self, candidates, model=None, device="cpu", symmetry_mode="none") -> dict[str, torch.Tensor]:
        """Drop-in replacement for ``build_embedding_cache`` that never re-runs the encoder."""
        paths = sorted({p for c in candidates for p in (c["img1"], c["img2"])}); self.get(paths)
        return {p: self.cache[str(p)].cpu() for p in paths}

    def install(self) -> None:
        """Route the pipeline's acquisition code through the cache."""
        pipeline.build_embedding_cache = self.embedding_cache


def fit_head(head: nn.Module, xa: torch.Tensor, xb: torch.Tensor, type_index: torch.Tensor, weight: torch.Tensor,
             winner: np.ndarray, reference_features: dict[str, torch.Tensor] | None, bad_features: torch.Tensor | None,
             lr: float, steps: int, weight_decay: float = 1e-4, anchor_weight: float = .25, bad_weight: float = .10) -> nn.Module:
    """Full-batch training on pre-computed features; invariant to the order of the rows."""
    head.eval()  # no dropout while training
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    index = torch.arange(len(type_index)); masks = {label: torch.as_tensor(winner == label) for label in LABELS}
    for _ in range(steps):
        a, b = head(xa)[index, type_index], head(xb)[index, type_index]
        terms = (-F.logsigmoid(a - b) * masks["1"] - F.logsigmoid(b - a) * masks["2"] + (a - b).abs() * masks["tie"]
                 + (F.relu(a) + F.relu(b)) * masks["not_apply"])
        loss = (terms * weight).sum() / max(len(type_index), 1)
        if reference_features and len(reference_features) > 1:
            scores = {name: head(f) for name, f in reference_features.items()}; values = []
            for preferred, sp in scores.items():
                column = TYPE_TO_INDEX[preferred]
                for other, so in scores.items():
                    if other != preferred: values.append(-F.logsigmoid(sp[:, column][:, None] - so[:, column][None, :]).mean())
            loss = loss + anchor_weight * torch.stack(values).mean()
        if bad_features is not None and len(bad_features): loss = loss + bad_weight * F.relu(head(bad_features) + 1.0).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
    return head.eval()


def train_model(exp: Experiment, features: FrozenFeatures, pair_ids: list[str], lr: float, steps: int,
                head_seed: int | None = None, anchor_weight: float = .25) -> BTModel:
    """Return a BTModel whose reward head was trained on ``pair_ids`` (order of ``pair_ids`` is irrelevant).

    ``head_seed`` re-initialises the head (default: the experiment seed's initialisation); ``anchor_weight`` scales the reference-ranking loss."""
    model = copy.deepcopy(features.base_model); rows = exp.rows_for(sorted(pair_ids))
    if head_seed is not None:
        torch.manual_seed(head_seed)
        for layer in model.reward_head:
            if hasattr(layer, "reset_parameters"): layer.reset_parameters()
    refs = {c: features.get(ps) for c, ps in exp.references.items()}
    bad = features.get(exp.bad_paths) if exp.bad_paths else None
    if rows.empty:  # no labelled pairs: only the reference and bad-image anchors train the head
        xa = xb = torch.empty(0, 512); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    else:
        xa, xb, typ = features.get(rows.resolved_img1), features.get(rows.resolved_img2), torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    fit_head(model.reward_head, xa, xb, typ, weight, winner, refs, bad, lr, steps, exp.cfg.weight_decay, anchor_weight=anchor_weight, bad_weight=exp.cfg.bad_anchor_weight)
    return model.eval()


@torch.no_grad()
def evaluate_model(exp: Experiment, features: FrozenFeatures, model: BTModel, split: str = "outer_test") -> dict:
    """Same win-rate-against-references rule as ``Experiment.evaluate``, computed on cached features."""
    images = exp.test_images if split == "outer_test" else exp.utility_images
    refs = {c: model.reward_head(features.get(ps)).numpy() for c, ps in exp.references.items()}
    correct = total = 0; by_class = {}
    for truth, paths in images.items():
        ok = 0
        for score in model.reward_head(features.get(paths)).numpy():
            win = {}
            for candidate in refs:
                opponents = np.concatenate([s for other, s in refs.items() if other != candidate]); idx = TYPE_TO_INDEX[candidate]
                win[candidate] = float(np.mean(1 / (1 + np.exp(-np.clip(score[idx] - opponents[:, idx], -50, 50)))))
            ok += max(win, key=win.get) == truth
        by_class[truth] = {"correct": int(ok), "total": len(paths), "accuracy": ok / len(paths)}; correct += ok; total += len(paths)
    return {"test_accuracy": correct / total, "test_correct": int(correct), "test_total": total, "by_class": by_class}


def save_feature_cache(cache: dict, path: Path, data_root: Path) -> None:
    """Persist cached features keyed by path relative to the data root, so the cache is machine independent."""
    root = str(Path(data_root).resolve()) + "/"
    torch.save({k[len(root):] if k.startswith(root) else k: v for k, v in cache.items()}, path)


def load_feature_cache(path: Path, data_root: Path) -> dict:
    root = str(Path(data_root).resolve()) + "/"
    return {root + k: v for k, v in torch.load(path).items()}


LABEL_FREE_BASES = ("uncertainty", "cluster_quota_uncertainty", "uncertainty_diversity", "cluster_margin_pairwise",
                    "mc_dropout_probability_variance", "mc_dropout_mutual_information")


def label_free_inputs(rows: list[dict], model: BTModel):
    """Make an original (type-aware) selector label-free: drop each candidate's type and silence the unused Twinned head.

    Without ``type_idx`` the original rules average over all five heads.  Zeroing the Twinned head's last-layer row makes its logit gap exactly 0
    for every pair (probability 0.5, no dropout variance), so it adds the same constant to every candidate and cannot change a ranking."""
    copy_ = copy.deepcopy(model); last = copy_.reward_head[3]
    with torch.no_grad(): last.weight[1].zero_(); last.bias[1].zero_()
    return [{k: v for k, v in r.items() if k != "type_idx"} for r in rows], copy_.eval()

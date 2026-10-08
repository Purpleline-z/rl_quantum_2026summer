"""Reward-model training on a frozen, cached encoder.

The encoder is applied once to every image and never updated.  Only the reward head is trained, by
full-batch gradient descent on summed per-row losses.  A sum over rows does not depend on how rows are
ordered, so the trained head is a function of the *set* of labelled pair groups (no minibatch order, no
dropout masks tied to row order).  Losses mirror ``Experiment.train``: Bradley--Terry for decisive
winners, an absolute-difference term for ties, a push-down term for ``not_apply``, reference-anchor
ranking, and bad-image negative anchors.
"""
from __future__ import annotations

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
from pairwise_active_learning_pipeline import Experiment, TYPE_TO_INDEX, transform  # noqa: E402


class FeatureStore:
    """Encodes each image path once with the experiment's initial (never-trained) encoder."""
    def __init__(self, exp: Experiment):
        self.model = exp.make_model().eval(); self.tf = transform(); self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def get(self, paths: Iterable) -> torch.Tensor:
        keys = [str(p) for p in paths]; missing = [k for k in dict.fromkeys(keys) if k not in self.cache]
        for i in range(0, len(missing), 16):
            chunk = missing[i:i + 16]
            batch = torch.stack([self.tf(Image.open(k).convert("L")) for k in chunk])
            for k, v in zip(chunk, self.model.encoder(batch)): self.cache[k] = v
        return torch.stack([self.cache[k] for k in keys]) if keys else torch.empty(0, 512)


def make_head(seed: int, hidden: int = 256, outputs: int = 5) -> nn.Module:
    generator = torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def _pair_loss(head, store: FeatureStore, exp: Experiment, pair_ids: list[str]) -> torch.Tensor:
    rows = exp.rows_for(sorted(pair_ids))
    xa, xb = head(store.get(rows.resolved_img1)), head(store.get(rows.resolved_img2))
    index = torch.arange(len(rows)); typ = torch.tensor(rows.type_idx.to_numpy())
    xa, xb = xa[index, typ], xb[index, typ]
    weight = torch.tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32); winner = rows.Winner.to_numpy()
    terms = torch.zeros(len(rows))
    for label in ("1", "2", "tie", "not_apply"):
        mask = torch.tensor(winner == label)
        if not mask.any(): continue
        term = (-F.logsigmoid(xa - xb) if label == "1" else -F.logsigmoid(xb - xa) if label == "2"
                else (xa - xb).abs() if label == "tie" else F.relu(xa) + F.relu(xb))
        terms = torch.where(mask, term, terms)
    return (terms * weight).sum() / len(rows)


def _anchor_loss(reference_scores: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean over class pairs and reference pairs of -log sigmoid(score_p(ref of p) - score_p(ref of o))."""
    values = []
    for preferred, scores_p in reference_scores.items():
        for other, scores_o in reference_scores.items():
            if other == preferred: continue
            column = TYPE_TO_INDEX[preferred]
            values.append(-F.logsigmoid(scores_p[:, column][:, None] - scores_o[:, column][None, :]).mean())
    return torch.stack(values).mean()


def train_head(exp: Experiment, store: FeatureStore, pair_ids: list[str], lr: float, steps: int, seed: int | None = None,
               anchor_weight: float = .25) -> nn.Module:
    head = make_head(exp.cfg.seed if seed is None else seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=exp.cfg.weight_decay)
    reference_features = {c: store.get(ps) for c, ps in exp.references.items()}
    bad = store.get(exp.bad_paths) if exp.bad_paths else None
    for _ in range(steps):
        loss = _pair_loss(head, store, exp, pair_ids)
        if len(reference_features) > 1:
            loss = loss + anchor_weight * _anchor_loss({c: head(f) for c, f in reference_features.items()})
        if bad is not None: loss = loss + exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
    return head.eval()


@torch.no_grad()
def evaluate_head(exp: Experiment, store: FeatureStore, head: nn.Module, split: str = "outer_test") -> dict:
    """Same win-rate-against-references rule as ``Experiment.evaluate``, on cached features."""
    images = exp.test_images if split == "outer_test" else exp.utility_images
    refs = {c: head(store.get(ps)).numpy() for c, ps in exp.references.items()}
    correct = total = 0; by_class = {}
    for truth, paths in images.items():
        ok = 0
        for score in head(store.get(paths)).numpy():
            win = {}
            for candidate in refs:
                opponents = np.concatenate([s for other, s in refs.items() if other != candidate]); idx = TYPE_TO_INDEX[candidate]
                win[candidate] = float(np.mean(1 / (1 + np.exp(-(score[idx] - opponents[:, idx])))))
            ok += max(win, key=win.get) == truth
        by_class[truth] = {"correct": ok, "total": len(paths), "accuracy": ok / len(paths)}; correct += ok; total += len(paths)
    return {"accuracy": correct / total, "correct": correct, "total": total, "by_class": by_class}

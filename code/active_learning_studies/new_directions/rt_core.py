"""Red-team audit toolkit (branch claude/redteam-*): per-judgment predictions and near-duplicate / novelty measures.

Image paths look like ``.../Trajectories/2022-02-04/115_RR220204A_441C_1738.bmp`` = <index>_<growth run>_<temperature>C_<frame>.  Features are the frozen SimCLR features of ``nd_core.feature_cache``;
similarities are cosine similarities of mean-centred, L2-normalised features (as in the graph).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nd_core as core
import nd_graph_learner as G
import nd_learners as L

REGISTRY = {**L.LEARNERS, **G.LEARNERS}

NAME = re.compile(r"/(\d+)_(RR\d+[A-Z]?)_(\d+)C?_(\d+)(?:-\d+)?\.bmp$")
AUDIT_SEEDS = tuple(range(8800, 8820)); CONFIRM_SEEDS = tuple(range(8850, 8885))


def session_of(p) -> str: return str(p).split("/")[-2]
def run_of(p):
    m = NAME.search(str(p)); return m.group(2) if m else None
def frame_of(p):
    m = NAME.search(str(p)); return int(m.group(4)) if m else None


class Sim:
    """Centred, L2-normalised features of every cached image (the geometry used by the kNN graph)."""
    def __init__(self, cache):
        self.paths = sorted(cache); self.index = {p: i for i, p in enumerate(self.paths)}
        x = torch.stack([cache[p].float() for p in self.paths]); x = x - x.mean(0); self.xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)

    def max_sim(self, queries, refs) -> np.ndarray:
        """For every query image the maximum cosine similarity to any of ``refs`` (excluding the identical image)."""
        r = sorted(set(refs)); qi = [self.index[q] for q in queries]; ri = torch.as_tensor([self.index[p] for p in r])
        s = self.xn[qi] @ self.xn[ri].T
        for row, q in enumerate(queries):
            same = [c for c, p in enumerate(r) if p == q]
            if same: s[row, same] = -2.0
        return s.max(1).values.numpy()


def heldout_images(ctx) -> set:
    return {x for g in list(ctx.validation) + list(ctx.test) for r in ctx.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}


def labelled_images(ctx, labelled) -> set:
    return {x for j in labelled for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}


def predict(ctx, name: str, labelled, seed: int, test) -> np.ndarray:
    return np.asarray(REGISTRY[name](ctx, labelled, seed)(test), dtype=float)


def novelty_table(sim: Sim, ctx, labelled, test, graph_nodes) -> pd.DataFrame:
    """Per test judgment: 1 - max cosine similarity of each of its images to the training-judgment images and to the graph nodes; and the min / max of the two images."""
    train = labelled_images(ctx, labelled); a, b = list(test.img1), list(test.img2); out = {}
    for tag, refs in (("train", train), ("node", graph_nodes)):
        na, nb = 1 - sim.max_sim(a, refs), 1 - sim.max_sim(b, refs); out[f"nov_{tag}_min"] = np.minimum(na, nb); out[f"nov_{tag}_max"] = np.maximum(na, nb)
    return pd.DataFrame(out)

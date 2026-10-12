"""Time-contrastive adapter on the frozen SimCLR features of the unlabelled trajectory frames.

Positives: frames of the same growth run whose frame indices differ by 1..3 (neighbouring frames share the surface state); negatives: other frames of the batch (other runs and distant times).
The adapter g: R^512 -> R^128 (linear, L2-normalised output) is trained with InfoNCE on ALL trajectory frames EXCEPT the images of the held-out validation/test groups of the seed
(so no held-out image is used, even without labels).  The head then sees [x, g(x)].  Usage: ``python3 nd_temporal.py build`` encodes every trajectory and ideal image once.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nd_core as core
import nd_learners as L

DATA = core.single_study.DATA


def build_all_features() -> None:
    cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(2000, "A", scratch, cache)
        paths = sorted(str(p.resolve()) for p in (DATA / "original data").glob("**/*") if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"))
        torch.set_num_threads(4); ctx.features.get(paths)
        core.frozen.save_feature_cache(ctx.features.cache, core.ALL_CACHE, DATA); print("encoded", len(ctx.features.cache), "images ->", core.ALL_CACHE)


def trajectory_index(paths):
    out = {}
    for p in paths:
        m = L._NAME.match(Path(p).name)
        if m: out[p] = (m.group(2), int(m.group(1)))
    return out


def train_adapter(ctx, held_out_images: set, seed: int, dim: int = 128, steps: int = 150, batch: int = 256, tau: float = .1, window: int = 3) -> nn.Module:
    info = trajectory_index([p for p in ctx.features.cache if "Trajectories" in p and p not in held_out_images])
    frames = sorted(info); rows = {p: i for i, p in enumerate(frames)}; by_run = {}
    for p in frames: by_run.setdefault(info[p][0], {})[info[p][1]] = p
    X = torch.stack([ctx.features.cache[p] for p in frames]); rng = np.random.default_rng(seed)
    torch.manual_seed(seed); g = nn.Linear(512, dim, bias=False); opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    for _ in range(steps):
        anchor = rng.choice(len(frames), size=min(batch, len(frames)), replace=False); pos = []
        for i in anchor:
            run, idx = info[frames[i]]; options = [by_run[run][idx + o] for o in range(-window, window + 1) if o != 0 and (idx + o) in by_run[run]]
            pos.append(rows[options[rng.integers(len(options))]] if options else i)
        za, zp = F.normalize(g(X[anchor]), dim=1), F.normalize(g(X[pos]), dim=1)
        logits = za @ zp.T / tau; loss = F.cross_entropy(logits, torch.arange(len(anchor)))
        opt.zero_grad(); loss.backward(); opt.step()
    return g.eval()


def timecontrast(ctx, labelled, seed, mode: str = "concat"):
    if "adapter" not in ctx.__dict__:
        held = {x for g in list(ctx.validation) + list(ctx.test) for r in ctx.exp.groups[g].itertuples() for x in (str(r.resolved_img1), str(r.resolved_img2))}
        ctx.adapter = train_adapter(ctx, held, ctx.seed)
    rms = float(torch.stack(list(ctx.features.cache.values())).pow(2).mean().sqrt())
    def fn(paths):
        x = ctx.features.get(paths)
        with torch.no_grad(): z = F.normalize(ctx.adapter(x), dim=1) * rms * (512 / 128) ** .5 * .5
        return torch.cat([x, z], dim=1) if mode == "concat" else z
    dim = 512 + 128 if mode == "concat" else 128; lr, steps = ctx.params_for(labelled)
    head = core.fit_generic(ctx, labelled, fn, dim, lr, steps, seed)
    return lambda test: core.head_d(head, fn, test)


LEARNERS = {"timecontrast": timecontrast, "timecontrast_only": lambda c, l, s: timecontrast(c, l, s, "only")}

if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    build_all_features()

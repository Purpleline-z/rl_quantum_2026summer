"""Audit 4: clean, independent re-implementation of aw8_all / lap3_k10 / lap3_k10_st4 from the written loss specification.

Shares with the main code only data access (judgment rows of a label set, the typed reference / ideal image lists, the per-budget lr/steps table, the cached features) and the metric function of
``nd_core.TestSet``.  The training loop, the loss terms, the anchor terms, the identity filter (own SHA-256 of the file bytes) and the kNN graph are written here from the specification:

  head      Linear(512,256) - ReLU - Dropout(0.2, switched off while training) - Linear(256,5); torch.manual_seed(seed) before construction; AdamW(lr, weight_decay=cfg.weight_decay); clip 1.0; full batch
  pair loss for judgment i of type k with head scores a, b of its two images:  w_i * [ -logsig(a-b) if image 1 won, -logsig(b-a) if image 2 won, |a-b| if tie, relu(a)+relu(b) if not_apply ] summed / n_judgments
  anchors   for every class c with typed images F_c (references; with all_ideal also utility and outer-test ideal images): mean over ordered class pairs (c, o) of mean_{x in F_c, y in F_o} -logsig(s_c(x) - s_c(y)); weight 8
  bad       0.10 * mean relu(score + 1) over the 'bad' images (all five outputs)
  Laplacian lap * mean over 256 random undirected kNN edges (i, j) of  sum_{k in {0,2,3,4}} (s_k(x_i) - s_k(x_j))^2  on the RAW features of the nodes (graph nodes = trajectory + ideal images minus the
            held-out groups' images and the outer-test ideal images, by SHA-256 of the file bytes; kNN = cosine top-10 on mean-centred, L2-normalised features, symmetrised)
"""
from __future__ import annotations

import argparse, hashlib, json, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nd_core as core  # sets sys.path for the repository modules
import judgment_unit_study as ju_study

TYPES = (0, 2, 3, 4)


def sha(path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Graph:
    def __init__(self, ctx, cache, k=10):
        held = {x for g in list(ctx.validation) + list(ctx.test) for r in ctx.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
        outer = {str(p) for ps in ctx.exp.test_images.values() for p in ps}; bad = {sha(p) for p in held | outer}
        self.paths = [p for p in sorted(cache) if sha(p) not in bad]; x = torch.stack([cache[p].float() for p in self.paths]); self.x = x
        xc = x - x.mean(0); xn = xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-9); s = xn @ xn.T; s.fill_diagonal_(-2.0); nb = s.topk(k, dim=1).indices; n = len(self.paths)
        a = torch.zeros(n, n, dtype=torch.bool); a[torch.arange(n)[:, None].expand_as(nb), nb] = True; a = a | a.T; r, c = torch.nonzero(a, as_tuple=True); keep = r < c; self.edges = torch.stack([r[keep], c[keep]], 1)


def fit(ctx, cache, labelled, seed, graph=None, lap=0.0, steps_mult=1.0, lr_mult=1.0, aw=8.0, all_ideal=True):
    rows = ju_study.rows_of(ctx.exp, labelled); lr, steps = ctx.params_for(labelled); lr, steps = lr * lr_mult, int(round(steps * steps_mult))
    feat = lambda ps: torch.stack([cache[str(p)].float() for p in ps])
    xa, xb = feat(list(rows.resolved_img1)), feat(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy()); w = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32); win = rows.Winner.to_numpy()
    torch.manual_seed(seed); head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 5)).eval()
    classes = {}
    for name, ps in ctx.exp.references.items():
        extra = (list(ctx.exp.utility_images.get(name, [])) + list(ctx.exp.test_images.get(name, []))) if all_ideal else []; classes[name] = (core.frozen.TYPE_TO_INDEX[name], feat(list(ps) + extra))
    bad = feat(ctx.exp.bad_paths) if ctx.exp.bad_paths else None; opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); idx = torch.arange(len(typ)); gen = torch.Generator().manual_seed(seed * 1000003 + 17)
    m1, m2, mt, mn = (torch.as_tensor(win == v) for v in ("1", "2", "tie", "not_apply"))
    for _ in range(steps):
        a, b = head(xa)[idx, typ], head(xb)[idx, typ]
        loss = (w * (-F.logsigmoid(a - b) * m1 - F.logsigmoid(b - a) * m2 + (a - b).abs() * mt + (F.relu(a) + F.relu(b)) * mn)).sum() / max(len(typ), 1)
        if len(classes) > 1:
            sc = {n: head(f) for n, (c, f) in classes.items()}; terms = []
            for n, (c, _) in classes.items():
                for o in classes:
                    if o != n: terms.append(-F.logsigmoid(sc[n][:, c][:, None] - sc[o][:, c][None, :]).mean())
            loss = loss + aw * torch.stack(terms).mean()
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0:
            e = graph.edges[torch.randint(len(graph.edges), (256,), generator=gen)]; si, sj = head(graph.x[e[:, 0]])[:, list(TYPES)], head(graph.x[e[:, 1]])[:, list(TYPES)]; loss = loss + lap * ((si - sj) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def scores(head, cache, test) -> np.ndarray:
    with torch.no_grad(): a = head(torch.stack([cache[str(p)].float() for p in test.img1])); b = head(torch.stack([cache[str(p)].float() for p in test.img2]))
    return (a - b)[torch.arange(len(test.k)), torch.as_tensor(test.k)].numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8819"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budgets", default="20,40"); ap.add_argument("--draws", type=int, default=2); args = ap.parse_args(); torch.set_num_threads(1); lo, hi = (args.seeds.split("-") * 2)[:2]; shard, n = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True); cache = {str(k): v for k, v in core.feature_cache().items()}
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(range(int(lo), int(hi) + 1)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, core.feature_cache()); test = core.TestSet(ctx); graph = Graph(ctx, cache); recs = []
                for budget in [int(b) for b in args.budgets.split(",")]:
                    for draw in range(args.draws):
                        lab = core.random_labelled(ctx, budget, draw)
                        for name, kw in (("aw8_all", {}), ("lap3_k10", {"lap": 3.0}), ("lap3_k10_st4", {"lap": 3.0, "steps_mult": 4.0}), ("aw8_all_st2lr0.5", {"steps_mult": 2.0, "lr_mult": 0.5})):
                            d = scores(fit(ctx, cache, lab, seed * 7 + draw, graph, **kw), cache, test); recs.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "d": [float(x) for x in d]})
                target.write_text(json.dumps(recs)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()

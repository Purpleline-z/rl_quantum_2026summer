#!/usr/bin/env python3
"""Runner for the new methods: judgment_unit_study with extra row rules registered (no change to the existing study code)."""
import judgment_unit_study as study
import judgment_unit_strategies as ju
import new_methods_strategies as nm

ju.ROW_RULES.update(nm.NEW); ju.STRATEGY_FAMILY = ju.STRATEGY_FAMILY + tuple(n for n in nm.NEW if n not in ju.STRATEGY_FAMILY)
_original_init = study.Context.__init__
def _init(self, *a, **k):
    _original_init(self, *a, **k); nm.CTX["ctx"] = self
study.Context.__init__ = _init

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
_original_evaluate = study.Context.evaluate
TYPE_NAMES = {0: "t1x1", 2: "tc6x2", 3: "t13", 4: "htr"}
def _evaluate(self, model):
    out = _original_evaluate(self, model)
    rows = self.exp.rows_for(self.test); rows = rows[rows.Winner.isin(["1", "2"])]; head = model.reward_head.eval()
    with torch.no_grad(): a, b = head(self.features.get(rows.resolved_img1)), head(self.features.get(rows.resolved_img2))
    d = (a - b)[torch.arange(len(rows)), torch.as_tensor(rows.type_idx.to_numpy())].numpy(); s = np.where((rows.Winner == "1").to_numpy(), 1.0, -1.0); typ = rows.type_idx.to_numpy()
    for k, name in TYPE_NAMES.items():
        m = typ == k; out[f"{name}_rows"] = int(m.sum())
        out[f"{name}_acc"] = float(((d[m] * s[m]) > 0).mean()) if m.any() else float("nan")
        out[f"{name}_ll"] = float(np.logaddexp(0.0, -s[m] * d[m]).mean()) if m.any() else float("nan")
        out[f"{name}_auc"] = float(roc_auc_score(s[m] > 0, d[m])) if m.any() and len(set(s[m])) == 2 else float("nan")
    return out
study.Context.evaluate = _evaluate


import os
import run_frozen_encoder_task3 as _harness
if os.environ.get("NM_INITIAL_GROUPS"):   # larger initial set (labels already in hand), e.g. 60 groups
    _make = _harness.make_experiment
    def _make_big(*a, **k):
        exp = _make(*a, **k); exp.cfg.initial_pairs = int(os.environ["NM_INITIAL_GROUPS"]); return exp
    _harness.make_experiment = _make_big


if os.environ.get("NM_HALF"):   # restrict the whole study to one of two image-disjoint halves of the 168 pair groups (independent data worlds)
    import random as _random
    _make_half_base = _harness.make_experiment
    def _make_half(*a, **k):
        exp = _make_half_base(*a, **k); _load = exp.load_and_split
        def load():
            initial, cands = _load(); ids = list(dict.fromkeys(list(initial) + list(cands))); imgs = {g: {exp.groups[g].iloc[0].resolved_img1, exp.groups[g].iloc[0].resolved_img2} for g in ids}
            parent = {g: g for g in ids}
            def find(x):
                while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
                return x
            owner = {}
            for g in sorted(ids):
                for im in sorted(imgs[g]):
                    if im in owner: parent[find(g)] = find(owner[im])
                    else: owner[im] = g
            comps = {}
            for g in sorted(ids): comps.setdefault(find(g), []).append(g)
            order = sorted(comps); _random.Random(2026).shuffle(order); half1, size = set(), 0
            for key in order:
                if size >= len(ids) // 2: break
                half1 |= set(comps[key]); size += len(comps[key])
            keep = half1 if os.environ["NM_HALF"] == "1" else set(ids) - half1
            ini = [g for g in initial if g in keep]; rest = [g for g in cands if g in keep]
            while len(ini) < 10 and rest: ini.append(rest.pop(0))
            return ini[:10], rest + ini[10:]
        exp.load_and_split = load; return exp
    _harness.make_experiment = _make_half


if os.environ.get("NM_CROSS"):   # train world = image-disjoint half NM_CROSS (initial set and pool); test set = ALL groups of the other half
    import random as _random2
    import pair_preference_endpoint as _endpoint
    _orig_split = _endpoint.split_classifier2_style
    def _halves(exp, ids):
        imgs = {g: {exp.groups[g].iloc[0].resolved_img1, exp.groups[g].iloc[0].resolved_img2} for g in ids}; parent = {g: g for g in ids}
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        owner = {}
        for g in sorted(ids):
            for im in sorted(imgs[g]):
                if im in owner: parent[find(g)] = find(owner[im])
                else: owner[im] = g
        comps = {}
        for g in sorted(ids): comps.setdefault(find(g), []).append(g)
        order = sorted(comps); _random2.Random(2026).shuffle(order); half1, size = set(), 0
        for key in order:
            if size >= len(ids) // 2: break
            half1 |= set(comps[key]); size += len(comps[key])
        return half1, set(ids) - half1
    def _cross_split(exp, initial, candidates, **kw):
        ids = list(dict.fromkeys(list(initial) + list(candidates))); h1, h2 = _halves(exp, ids); train, test = (h1, h2) if os.environ["NM_CROSS"] == "1" else (h2, h1)
        tr = [g for g in ids if g in train]; ini, pool, _, _ = _orig_split(exp, tr[:10], tr[10:], test_fraction=0.0)
        return ini, pool, [], [g for g in ids if g in test]
    _endpoint.split_classifier2_style = _cross_split


if __name__ == "__main__":
    study.main()

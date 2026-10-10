#!/usr/bin/env python3
"""Post-hoc pooled summary of the frozen typed_decisive_coverage rule over all fresh confirmatory seed sets run under the re-tuned schedule: P1 (1500-1529), P3 (1400-1434), P5 (1600-1639).
Controls (type-only, shuffled posterior) exist for P1 and P5 only.  Seed-level Wilcoxon on the pooled per-seed gains; Holm over the four blocks; raw p for the paired differences."""
import numpy as np
import pandas as pd

import aggregate_overnight2 as a

RULE = "typed_decisive_coverage"
print(f"{'block':14s} {'n':>4s} {'log-loss':>9s} {'raw p':>7s} {'Holm4':>7s} {'AUC':>8s} {'acc':>8s} {'better':>7s} | minus typeonly (n) LL / AUC p | minus shuffled LL / AUC p | minus vopt_u (n) LL p")
rows = {}
for split, cond in a.BLOCKS:
    gains = {m: pd.concat([a.gains(d, split, cond, m)[[RULE]] for d in ("graph_p1", "graph_p3", "graph_p5")]) for m in a.METRICS}
    ll = gains["test_decisive_log_loss"][RULE].dropna().to_numpy(); rows[(split, cond)] = ll
    ctl = {}
    for ctrl in ("typed_decisive_coverage_typeonly", "typed_decisive_coverage_shuffled", "vopt_u"):
        parts = {m: pd.concat([a.gains(d, split, cond, m)[[RULE, ctrl]].dropna() for d in ("graph_p1", "graph_p5")] if ctrl != "vopt_u" else [a.gains(d, split, cond, m)[[RULE, ctrl]].dropna() for d in ("graph_p1", "graph_p3", "graph_p5")]) for m in ("test_decisive_log_loss", "test_decisive_auc")}
        dl = (parts["test_decisive_log_loss"][RULE] - parts["test_decisive_log_loss"][ctrl]).to_numpy(); da = (parts["test_decisive_auc"][RULE] - parts["test_decisive_auc"][ctrl]).to_numpy()
        ctl[ctrl] = (len(dl), dl.mean(), a.wp(dl), da.mean(), a.wp(da))
    rows[(split, cond)] = (ll, gains["test_decisive_auc"][RULE].mean(), gains["test_decisive_accuracy"][RULE].mean(), ctl)
pvals = {k: a.wp(v[0]) for k, v in rows.items()}; hp = a.holm(pvals)
for (split, cond), (ll, au, ac, ctl) in rows.items():
    t, s, v = ctl["typed_decisive_coverage_typeonly"], ctl["typed_decisive_coverage_shuffled"], ctl["vopt_u"]
    print(f"{split}-{cond:10s} {len(ll):4d} {ll.mean():+9.3f} {a.fp(pvals[(split, cond)]):>7s} {a.fp(hp[(split, cond)]):>7s} {au:+8.4f} {ac:+8.4f} {(ll > 0).mean():7.0%} | {t[1]:+.3f}/{t[3]:+.4f} p {a.fp(t[2])}/{a.fp(t[4])} (n={t[0]}) | {s[1]:+.3f}/{s[3]:+.4f} p {a.fp(s[2])}/{a.fp(s[4])} | {v[1]:+.3f} (n={v[0]}) p {a.fp(v[2])}")

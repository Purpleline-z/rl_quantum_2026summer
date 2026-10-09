#!/usr/bin/env python3
"""Round 4 replication (graph_exploration/AUTONOMOUS_RUN_LOG.md): split B, fresh seeds 430-459.  H1: typed_decisive_coverage gain over Random (Holm over the two conditions);
H2: real minus shuffled / minus type-only, paired over seeds (raw p, uncorrected)."""
import numpy as np
import pandas as pd

import aggregate_graph_vs_random as agg

SEEDS = set(range(430, 460)); RULE = "typed_decisive_coverage"; OTHERS = [RULE + "_typeonly", RULE + "_shuffled", "core_set_relation"]
fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
results = {}
for cond in ("single", "sequential"):
    cells = agg.load("B", cond); cells = cells[cells.seed.isin(SEEDS)]
    for metric, (label, lower) in agg.METRICS.items():
        sign = -1.0 if lower else 1.0; per = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
        rnd = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); p = per.merge(rnd, on=["seed", "budget"]); p["gain"] = sign * (p[metric] - p.random)
        results[(cond, metric)] = p[p.family != "random"].groupby(["family", "seed"]).gain.mean().unstack(0)
for cond in ("single", "sequential"):
    ll, au = results[(cond, "test_decisive_log_loss")], results[(cond, "test_decisive_auc")]
    print(f"\nsplit B, {cond}, seeds 430-459 (n per rule: {ll.count().to_dict()})")
    for f in [RULE] + OTHERS:
        if f in ll: v = ll[f].dropna().to_numpy(); print(f"  {f:40s} log-loss gain {v.mean():+.3f} (raw p {fp(agg.wilcoxon_p(v))}, better {(v > 0).mean():.0%})  AUC gain {au[f].dropna().mean():+.4f}")
    for f in OTHERS:
        if RULE in ll and f in ll: d = (ll[RULE] - ll[f]).dropna().to_numpy(); print(f"    {RULE} minus {f}: {d.mean():+.3f} (raw p {fp(agg.wilcoxon_p(d))}, n={len(d)})")
p1 = {c: agg.wilcoxon_p(results[(c, 'test_decisive_log_loss')][RULE].dropna().to_numpy()) for c in ("single", "sequential") if RULE in results[(c, 'test_decisive_log_loss')]}
print("\nH1 Holm over the two conditions:", {c: fp(v) for c, v in agg.holm({k: v for k, v in p1.items() if not np.isnan(v)}).items()})


def markdown() -> str:
    rows = ["| Condition | Rule | log-loss gain | raw p | AUC gain | seeds better |", "|---|---|---:|---:|---:|---:|"]
    names = {RULE: "Type-aware graph coverage x P(decisive)", RULE + "_typeonly": "(control) decisive predictor from the type only", RULE + "_shuffled": "(control) type posterior shuffled over images", "core_set_relation": "(baseline) Core-set, relation-aware pairs"}
    for cond in ("single", "sequential"):
        ll, au = results[(cond, "test_decisive_log_loss")], results[(cond, "test_decisive_auc")]
        for f in [RULE] + OTHERS:
            v = ll[f].dropna().to_numpy(); rows.append(f"| {'single-shot' if cond == 'single' else 'sequential'} | {names[f]} | {v.mean():+.3f} | {fp(agg.wilcoxon_p(v))} | {au[f].dropna().mean():+.4f} | {(v > 0).mean():.0%} ({len(v)}) |")
    diffs = []
    for cond in ("single", "sequential"):
        ll = results[(cond, "test_decisive_log_loss")]
        for f, nm in [(RULE + "_typeonly", "type-only control"), (RULE + "_shuffled", "shuffled-posterior control"), ("core_set_relation", "Core-set baseline")]:
            d = (ll[RULE] - ll[f]).dropna().to_numpy(); diffs.append(f"| {'single-shot' if cond == 'single' else 'sequential'} | rule minus {nm} | {d.mean():+.3f} | {fp(agg.wilcoxon_p(d))} |")
    holm = agg.holm({k: v for k, v in p1.items() if not np.isnan(v)})
    return "\n".join(rows) + "\n\n| Condition | Paired difference (log-loss) | mean | raw p |\n|---|---|---:|---:|\n" + "\n".join(diffs) + f"\n\nH1 (gain over Random, Holm over the two conditions): single-shot p = {fp(holm['single'])}, sequential p = {fp(holm['sequential'])}.\n"


#!/usr/bin/env python3
"""Confirmation analysis of the frozen candidates (see graph_exploration/AUTONOMOUS_RUN_LOG.md): CONFIRM seeds 410-429 and 42, 79, 123, 202, 303.

Per block (split x condition): gain over Random (mean of 5 draws, per seed averaged over budgets 10/20/40/60); seed-level Wilcoxon; Holm over the CANDIDATES only.
Controls and embedding baselines are reported with raw p (not part of the Holm family).  Success criterion (fixed in the log): a candidate with Holm p < 0.05 for log-loss in >= 2 blocks
including both splits, and a non-negative AUC gain in those blocks.
usage: aggregate_confirm.py  [--markdown]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import aggregate_graph_vs_random as agg

CONFIRM = set(range(410, 430)) | {42, 79, 123, 202, 303}
CANDIDATES = ["typed_decisive_coverage_unc", "typed_decisive_coverage", "typed_decisive_bald"]
CONTROLS = ["typed_decisive_coverage_unc_typeonly", "typed_decisive_coverage_unc_shuffled"]
BASELINES = ["core_set_relation", "bald_decisive", "typiclust_pairs", "laplace_bald", "uncertainty"]
LABEL = {"typed_decisive_coverage_unc": "Type-aware graph coverage x P(decisive) x uncertainty", "typed_decisive_coverage": "Type-aware graph coverage x P(decisive)",
         "typed_decisive_bald": "Laplace BALD x P(decisive), graph features", "typed_decisive_coverage_unc_typeonly": "(control) as first, decisive predictor from type only",
         "typed_decisive_coverage_unc_shuffled": "(control) as first, type posterior shuffled over images", "core_set_relation": "(baseline) Core-set, relation-aware pairs",
         "bald_decisive": "(baseline) BALD x P(decisive), pair-distance features", "typiclust_pairs": "(baseline) TypiClust (pairs)", "laplace_bald": "(baseline) Laplace BALD",
         "uncertainty": "(baseline) Uncertainty, own head"}
BLOCKS = [("A", "single"), ("A", "sequential"), ("B", "single"), ("B", "sequential")]


def block_gains(split: str, condition: str, metric: str):
    cells = agg.load(split, condition); cells = cells[cells.seed.isin(CONFIRM)]; lower = agg.METRICS[metric][1]; sign = -1.0 if lower else 1.0
    per = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    random = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); paired = per.merge(random, on=["seed", "budget"]); paired["gain"] = sign * (paired[metric] - paired.random)
    return paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index(), paired


def fp(p: float) -> str:
    return "n/a" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def main(markdown: bool) -> None:
    summary = {}; lines = []
    head = "| Split | Condition | Rule | log-loss gain | p (Holm for candidates) | AUC gain | seeds better | n |" if markdown else ""
    if markdown: lines += [head, "|---|---|---|---:|---:|---:|---:|---:|"]
    for split, condition in BLOCKS:
        sl, paired = block_gains(split, condition, "test_decisive_log_loss"); sa, _ = block_gains(split, condition, "test_decisive_auc")
        tests = {f: agg.wilcoxon_p(g.gain.to_numpy()) for f, g in sl.groupby("family")}; holm = agg.holm({k: v for k, v in tests.items() if k in CANDIDATES and not np.isnan(v)})
        for f in CANDIDATES + CONTROLS + BASELINES:
            g = sl[sl.family == f].gain.to_numpy(); a = sa[sa.family == f].gain.to_numpy()
            if len(g) == 0: continue
            p = holm.get(f, tests[f]); summary[(split, condition, f)] = (g.mean(), p, a.mean() if len(a) else np.nan, f in holm, len(g))
            row = f"| {split} | {'single-shot' if condition == 'single' else 'sequential'} | {LABEL[f]} | {g.mean():+.3f} | {fp(p)}{'' if f in holm else ' (raw)'} | {a.mean():+.4f} | {(g > 0).mean():.0%} | {len(g)} |"
            lines.append(row) if markdown else print(f"{split}-{condition[:3]:4s} {f:42s} logloss {g.mean():+.3f} p={fp(p):>7s}{'' if f in holm else '(raw)'} AUC {a.mean():+.4f} better {(g > 0).mean():.0%} n={len(g)}")
        if not markdown:
            wide = sl.pivot(index="seed", columns="family", values="gain")
            for c in CANDIDATES[:1]:
                for ctrl in CONTROLS:
                    if c in wide and ctrl in wide: d = (wide[c] - wide[ctrl]).dropna().to_numpy(); print(f"    {c} minus {ctrl}: {d.mean():+.3f} (raw p {fp(agg.wilcoxon_p(d))}, n={len(d)})")
            print()
    # success criterion
    print("\nSuccess criterion (Holm p < 0.05 for log-loss in >= 2 blocks incl. both splits, AUC gain >= 0 there):")
    for c in CANDIDATES:
        ok = [(s, cnd) for s, cnd in BLOCKS if summary.get((s, cnd, c)) and summary[(s, cnd, c)][1] < .05 and summary[(s, cnd, c)][0] > 0 and summary[(s, cnd, c)][2] >= 0]
        passed = len(ok) >= 2 and {"A", "B"} <= {s for s, _ in ok}
        print(f"  {c}: significant blocks {ok} -> {'MET' if passed else 'not met'}")
    if markdown: print("\n".join(lines))


if __name__ == "__main__":
    main("--markdown" in sys.argv)

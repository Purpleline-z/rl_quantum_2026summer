#!/usr/bin/env python3
"""Phase-2 replication test (AUTONOMOUS_RUN_LOG.md, "FROZEN PHASE 2 TEST"): seeds 430-449, candidate typed_decisive_coverage, Holm over its 4 blocks;
controls (typeonly, shuffled) and non-graph coverage references (core_set_relation, typiclust_pairs) reported with raw p. usage: aggregate_phase2.py [--out FILE.md]"""
from __future__ import annotations

import argparse

import numpy as np

import aggregate_confirm as ac
import aggregate_graph_vs_random as agg

ac.SEEDS = list(range(430, 450))
CAND = "typed_decisive_coverage"; OTHERS = ["typed_decisive_coverage_typeonly", "typed_decisive_coverage_shuffled", "core_set_relation", "typiclust_pairs"]; BLOCKS = ac.BLOCKS


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); out = ap.parse_args().out; lines: list[str] = []
    def emit(s: str = "") -> None: print(s); lines.append(s)
    res = {}
    for metric, (label, lower) in agg.METRICS.items():
        wides = {b: ac.seed_gains(*b, metric, lower) for b in BLOCKS}; stats = {}; raw = {}
        for b in BLOCKS:
            for name in [CAND] + OTHERS:
                if name not in wides[b]: continue
                v = wides[b][name].dropna().to_numpy(); lo, hi = ac.boot_ci(v); stats[(name, b)] = dict(mean=v.mean(), lo=lo, hi=hi, better=(v > 0).mean(), p=agg.wilcoxon_p(v), n=len(v))
                if name == CAND: raw[b] = stats[(name, b)]["p"]
        h = agg.holm({k: v for k, v in raw.items() if not np.isnan(v)})
        for b, v in h.items(): stats[(CAND, b)]["holm"] = v
        res[label] = stats
        emit(f"\n#### {label}: gain over Random, seeds 430-449 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over the 4 blocks for the candidate))\n")
        emit("| rule | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for name in [CAND] + OTHERS:
            row = []
            for b in BLOCKS:
                s = stats.get((name, b)); hh = f", Holm {s['holm']:.3f}" if s and name == CAND else ""
                row.append("-" if s is None else f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] ({s['better']:.0%}, p {s['p']:.3f}{hh})")
            emit(f"| {name}{'' if name == CAND else ' (control/reference)'} | " + " | ".join(row) + " |")
        emit(f"\nn seeds per cell: {sorted({s['n'] for s in stats.values()})}")
    emit("\n#### Paired differences, log-loss (positive = coverage rule is better than the comparison)\n")
    emit("| coverage minus | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
    wides = {b: ac.seed_gains(*b, "test_decisive_log_loss", True) for b in BLOCKS}
    for other in OTHERS:
        row = []
        for b in BLOCKS:
            w = wides[b]
            if CAND in w and other in w: d = (w[CAND] - w[other]).dropna().to_numpy(); lo, hi = ac.boot_ci(d); row.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (p {agg.wilcoxon_p(d):.3f})")
            else: row.append("-")
        emit(f"| {other} | " + " | ".join(row) + " |")
    ll, au = res["log-loss"], res["AUC"]
    ok = all(ll.get((CAND, b), {}).get("holm", 1) < .05 and ll[(CAND, b)]["mean"] > 0 and au[(CAND, b)]["mean"] >= 0 for b in [("B", "single"), ("B", "sequential")])
    emit(f"\nReplication criterion (Holm p < 0.05, positive log-loss gain and AUC gain >= 0 in both B-single and B-sequential): {'MET' if ok else 'NOT met'}")
    if out: open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

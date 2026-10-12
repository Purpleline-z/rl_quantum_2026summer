#!/usr/bin/env python3
"""Phase-3 frozen test (AUTONOMOUS_RUN_LOG.md, "FROZEN PHASE 3 TEST"): seeds 460-479; 3 candidates; Holm over 3 x 4 blocks per metric (log-loss, AUC);
controls (flat coverage, shuffled posterior) and the phase-1 rule reported with raw p.  Success: Holm p < .05 and positive gain in BOTH metrics in >= 2 blocks incl. both splits.
usage: aggregate_phase3.py [--out FILE.md]"""
from __future__ import annotations

import argparse

import numpy as np

import aggregate_confirm_session2 as ac
import aggregate_graph_vs_random as agg

ac.SEEDS = list(range(460, 480)); BLOCKS = ac.BLOCKS
CAND = ["typed_decisive_coverage_ul_k10a99", "typed_zeroshot_coverage_a9", "typed_zeroshot_coverage_unc"]
OTHERS = ["typed_flat_coverage_a99", "typed_flat_coverage_ul_k10", "typed_decisive_coverage_ul_k10a99_shuffled", "typed_zeroshot_coverage_a9_shuffled", "typed_zeroshot_coverage_unc_shuffled", "typed_decisive_coverage"]
PAIRS = [("typed_decisive_coverage_ul_k10a99", "typed_decisive_coverage_ul_k10a99_shuffled"), ("typed_decisive_coverage_ul_k10a99", "typed_flat_coverage_ul_k10"),
         ("typed_zeroshot_coverage_a9", "typed_zeroshot_coverage_a9_shuffled"), ("typed_zeroshot_coverage_a9", "typed_flat_coverage_a99"),
         ("typed_zeroshot_coverage_unc", "typed_zeroshot_coverage_unc_shuffled"), ("typed_zeroshot_coverage_unc", "typed_flat_coverage_a99")]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); out = ap.parse_args().out; lines: list[str] = []
    def emit(s: str = "") -> None: print(s); lines.append(s)
    res = {}; wides = {}
    for metric, (label, lower) in agg.METRICS.items():
        wides[label] = {b: ac.seed_gains(*b, metric, lower) for b in BLOCKS}; stats = {}; raw = {}
        for b in BLOCKS:
            for name in CAND + OTHERS:
                if name not in wides[label][b]: continue
                v = wides[label][b][name].dropna().to_numpy(); lo, hi = ac.boot_ci(v); stats[(name, b)] = dict(mean=v.mean(), lo=lo, hi=hi, better=(v > 0).mean(), p=agg.wilcoxon_p(v), n=len(v))
                if name in CAND: raw[(name, b)] = stats[(name, b)]["p"]
        h = agg.holm({k: v for k, v in raw.items() if not np.isnan(v)})
        for k, v in h.items(): stats[k]["holm"] = v
        res[label] = stats
        emit(f"\n#### {label}: gain over Random, seeds 460-479 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over 12 candidate tests))\n")
        emit("| rule | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for name in CAND + OTHERS:
            row = []
            for b in BLOCKS:
                s = stats.get((name, b)); hh = f", Holm {s['holm']:.3f}" if s and name in CAND else ""
                row.append("-" if s is None else f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] ({s['better']:.0%}, p {s['p']:.3f}{hh})")
            emit(f"| {name}{'' if name in CAND else ' (control/reference)'} | " + " | ".join(row) + " |")
        emit(f"\nn seeds per cell: {sorted({s['n'] for s in stats.values()})}")
    for label in ("log-loss", "AUC"):
        emit(f"\n#### Paired differences, {label} gain (candidate minus control; positive = the candidate's ingredient helps)\n")
        emit("| candidate minus control | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for a, c in PAIRS:
            row = []
            for b in BLOCKS:
                w = wides[label][b]
                if a in w and c in w: d = (w[a] - w[c]).dropna().to_numpy(); lo, hi = ac.boot_ci(d); row.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (p {agg.wilcoxon_p(d):.3f})")
                else: row.append("-")
            emit(f"| {a.replace('typed_', '')} minus {c.replace('typed_', '')} | " + " | ".join(row) + " |")
    emit("\n#### Success criterion (Holm p < .05 and positive gain in BOTH log-loss and AUC in >= 2 blocks incl. both splits)\n")
    for name in CAND:
        ok = [b for b in BLOCKS if all(res[l].get((name, b), {}).get("holm", 1) < .05 and res[l][(name, b)]["mean"] > 0 for l in ("log-loss", "AUC"))]
        passed = len(ok) >= 2 and {b[0] for b in ok} == {"A", "B"}; emit(f"- {name}: blocks significant in both metrics {[f'{s}-{c}' for s, c in ok]} -> {'MET' if passed else 'NOT met'}")
    if out: open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase-4 frozen replication (AUTONOMOUS_RUN_LOG.md): seeds 480-499, single candidate typed_zeroshot_coverage_unc, PRIMARY metric AUC gain over Random, Holm over its 4 blocks;
log-loss secondary; controls flat coverage and shuffled posterior. usage: aggregate_phase4.py [--out FILE.md]"""
from __future__ import annotations

import argparse

import numpy as np

import aggregate_confirm_session2 as ac
import aggregate_graph_vs_random as agg

ac.SEEDS = list(range(480, 500)); BLOCKS = ac.BLOCKS
CAND = "typed_zeroshot_coverage_unc"; CTRL = ["typed_flat_coverage_a99", "typed_zeroshot_coverage_unc_shuffled"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); out = ap.parse_args().out; lines: list[str] = []
    def emit(s: str = "") -> None: print(s); lines.append(s)
    res = {}; wide = {}
    for metric, (label, lower) in agg.METRICS.items():
        wide[label] = {b: ac.seed_gains(*b, metric, lower) for b in BLOCKS}; stats = {}
        for b in BLOCKS:
            for name in [CAND] + CTRL:
                v = wide[label][b][name].dropna().to_numpy(); lo, hi = ac.boot_ci(v); stats[(name, b)] = dict(mean=v.mean(), lo=lo, hi=hi, better=(v > 0).mean(), p=agg.wilcoxon_p(v), n=len(v))
        h = agg.holm({(CAND, b): stats[(CAND, b)]["p"] for b in BLOCKS})
        for k, v in h.items(): stats[k]["holm"] = v
        res[label] = stats
        emit(f"\n#### {label}{' (PRIMARY)' if label == 'AUC' else ' (secondary)'}: gain over Random, seeds 480-499 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over the 4 blocks for the candidate))\n")
        emit("| rule | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for name in [CAND] + CTRL:
            row = []
            for b in BLOCKS:
                s = stats[(name, b)]; hh = f", Holm {s['holm']:.3f}" if name == CAND else ""
                row.append(f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] ({s['better']:.0%}, p {s['p']:.3f}{hh})")
            emit(f"| {name}{'' if name == CAND else ' (control)'} | " + " | ".join(row) + " |")
        emit(f"\nn seeds per cell: {sorted({s['n'] for s in stats.values()})}")
    for label in ("AUC", "log-loss"):
        emit(f"\n#### Paired differences, {label} gain (candidate minus control)\n"); emit("| candidate minus | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for c in CTRL:
            row = []
            for b in BLOCKS:
                d = (wide[label][b][CAND] - wide[label][b][c]).dropna().to_numpy(); lo, hi = ac.boot_ci(d); row.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (p {agg.wilcoxon_p(d):.3f})")
            emit(f"| {c.replace('typed_', '')} | " + " | ".join(row) + " |")
    ok = [b for b in BLOCKS if res["AUC"][(CAND, b)]["holm"] < .05 and res["AUC"][(CAND, b)]["mean"] > 0]; passed = len(ok) >= 2 and {b[0] for b in ok} == {"A", "B"}
    emit(f"\nReplication criterion (AUC: Holm p < .05 and positive in >= 2 blocks incl. both splits): significant blocks {[f'{s}-{c}' for s, c in ok]} -> {'MET' if passed else 'NOT met'}")
    if out: open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

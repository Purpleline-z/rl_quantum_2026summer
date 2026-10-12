#!/usr/bin/env python3
"""Phase-6 reference comparison (AUTONOMOUS_RUN_LOG.md): seeds 500-519; typed_zeroshot_coverage_unc (graph-aware) against four non-graph embedding baselines on the same seeds.
Paired differences in AUC (primary) and log-loss gain; raw Wilcoxon p and Holm over the 16 comparisons per metric. usage: aggregate_phase6.py [--out FILE.md]"""
from __future__ import annotations

import argparse

import numpy as np

import aggregate_confirm_session2 as ac
import aggregate_graph_vs_random as agg

ac.SEEDS = list(range(500, 520)); BLOCKS = ac.BLOCKS; C = "typed_zeroshot_coverage_unc"; BASE = ["image_coverage_uncertainty", "core_set_relation", "bald_decisive", "laplace_bald"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); out = ap.parse_args().out; lines: list[str] = []
    def emit(s: str = "") -> None: print(s); lines.append(s)
    f = lambda d: f"{d.mean():+.3f} [{ac.boot_ci(d)[0]:+.3f}, {ac.boot_ci(d)[1]:+.3f}]"
    for metric, (label, lower) in agg.METRICS.items():
        wide = {b: ac.seed_gains(*b, metric, lower) for b in BLOCKS}
        emit(f"\n#### {label}: gain over Random, seeds 500-519 (mean [95% bootstrap CI], share of seeds better)\n"); emit("| rule | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for name in [C] + BASE: emit(f"| {name} | " + " | ".join(f"{f(wide[b][name].dropna().to_numpy())} ({(wide[b][name].dropna() > 0).mean():.0%})" for b in BLOCKS) + " |")
        diffs = {(n, b): (wide[b][C] - wide[b][n]).dropna().to_numpy() for n in BASE for b in BLOCKS}; p = {k: agg.wilcoxon_p(v) for k, v in diffs.items()}; h = agg.holm({k: v for k, v in p.items() if not np.isnan(v)})
        emit(f"\nPaired difference ({C} minus baseline), {label} (raw p; Holm over 16)\n"); emit("| baseline | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for n in BASE: emit(f"| {n} | " + " | ".join(f"{f(diffs[(n, b)])} (p {p[(n, b)]:.3f}, Holm {h[(n, b)]:.3f})" for b in BLOCKS) + " |")
    if out: open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

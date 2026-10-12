#!/usr/bin/env python3
"""Phase-5 frozen test (AUTONOMOUS_RUN_LOG.md): seeds 500-519. PRIMARY: paired AUC difference typed_zeroshot_coverage_unc minus typed_flat_unc_coverage_a99 (does the answerability prior add
ranking quality over coverage x own-head uncertainty?), Wilcoxon, Holm over 4 blocks; success = Holm < .05 and positive in >= 2 blocks.  Secondary: gain over Random (both rules), log-loss.
usage: aggregate_phase5.py [--out FILE.md]"""
from __future__ import annotations

import argparse

import numpy as np

import aggregate_confirm_session2 as ac
import aggregate_graph_vs_random as agg

ac.SEEDS = list(range(500, 520)); BLOCKS = ac.BLOCKS; A = "typed_zeroshot_coverage_unc"; B = "typed_flat_unc_coverage_a99"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); out = ap.parse_args().out; lines: list[str] = []
    def emit(s: str = "") -> None: print(s); lines.append(s)
    wide = {l: {b: ac.seed_gains(*b, m, lw) for b in BLOCKS} for m, (l, lw) in agg.METRICS.items()}
    f = lambda d: f"{d.mean():+.3f} [{ac.boot_ci(d)[0]:+.3f}, {ac.boot_ci(d)[1]:+.3f}] ({(d > 0).mean():.0%}, p {agg.wilcoxon_p(d):.3f})"
    for label in ("AUC", "log-loss"):
        emit(f"\n#### {label}: gain over Random, seeds 500-519 (mean [95% bootstrap CI] (share of seeds better, raw p))\n"); emit("| rule | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        for name in (A, B): emit(f"| {name} | " + " | ".join(f(wide[label][b][name].dropna().to_numpy()) for b in BLOCKS) + " |")
        emit(f"\nPaired difference {label} ({A} minus {B}):\n"); emit("| | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * 4)
        ds = {b: (wide[label][b][A] - wide[label][b][B]).dropna().to_numpy() for b in BLOCKS}
        h = agg.holm({b: agg.wilcoxon_p(d) for b, d in ds.items()}); emit("| difference | " + " | ".join(f(ds[b]) + f", Holm {h[b]:.3f}" for b in BLOCKS) + " |")
        if label == "AUC": ok = [b for b in BLOCKS if h[b] < .05 and ds[b].mean() > 0]
    emit(f"\nPrimary criterion (paired AUC difference, Holm < .05 and positive in >= 2 of 4 blocks): significant blocks {[f'{s}-{c}' for s, c in ok]} -> {'MET' if len(ok) >= 2 else 'NOT met'}")
    if out: open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Confirmation analysis of the frozen type-aware graph candidates (AUTONOMOUS_RUN_LOG.md, "FROZEN CANDIDATES").

usage: aggregate_confirm.py [--out FILE.md]
Seeds: 410-429 and 42, 79, 123, 202, 303 (25 seeds), used once. Per seed: gain over Random (mean of the Random draws) averaged over budgets 10/20/40/60.
Holm family (fixed before the run): the 3 candidates x 4 blocks (split x condition), separately for log-loss and AUC (12 tests each).
Controls (typeonly, shuffled) are reported with raw p and are NOT in the family. Also: real minus control paired differences and 95% paired-bootstrap intervals.
Success criterion: a candidate with Holm p < 0.05 for log-loss gain in >= 2 of the 4 blocks, including both splits, and a non-negative mean AUC gain in those blocks.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import aggregate_graph_vs_random as agg

SEEDS = list(range(410, 430)) + [42, 79, 123, 202, 303]
CANDIDATES = ["typed_decisive_coverage_unc", "typed_decisive_coverage", "typed_decisive_bald"]
CONTROLS = ["typed_decisive_coverage_unc_typeonly", "typed_decisive_coverage_unc_shuffled"]
BLOCKS = [("A", "single"), ("A", "sequential"), ("B", "single"), ("B", "sequential")]
rng = np.random.default_rng(0)


def boot_ci(v: np.ndarray, n: int = 10000) -> tuple[float, float]:
    idx = rng.integers(0, len(v), size=(n, len(v))); m = v[idx].mean(axis=1); return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def seed_gains(split: str, cond: str, metric: str, lower: bool) -> pd.DataFrame:
    cells = agg.load(split, cond); cells = cells[cells.seed.isin(SEEDS)]; sign = -1.0 if lower else 1.0
    per = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    random = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); paired = per.merge(random, on=["seed", "budget"]); paired["gain"] = sign * (paired[metric] - paired.random)
    return paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().unstack(0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); out = ap.parse_args().out; lines: list[str] = []
    def emit(s: str = "") -> None: print(s); lines.append(s)
    results = {}
    for metric, (label, lower) in agg.METRICS.items():
        raw = {}; stats = {}
        for split, cond in BLOCKS:
            wide = seed_gains(split, cond, metric, lower)
            for name in CANDIDATES + CONTROLS:
                if name not in wide: continue
                v = wide[name].dropna().to_numpy(); lo, hi = boot_ci(v); stats[(name, split, cond)] = dict(mean=v.mean(), sd=v.std(ddof=1), n=len(v), better=(v > 0).mean(), lo=lo, hi=hi, p=agg.wilcoxon_p(v))
                if name in CANDIDATES: raw[(name, split, cond)] = stats[(name, split, cond)]["p"]
        holm = agg.holm({k: v for k, v in raw.items() if not np.isnan(v)})
        for k, s in stats.items(): s["holm"] = holm.get(k, float("nan"))
        results[label] = stats
        emit(f"\n#### {label}: gain over Random, 25 confirmation seeds (mean [95% bootstrap CI], share of seeds better, raw Wilcoxon p, Holm p over 12 candidate tests)\n")
        emit("| rule | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * len(BLOCKS))
        for name in CANDIDATES + CONTROLS:
            cells = []
            for split, cond in BLOCKS:
                s = stats.get((name, split, cond)); h = f", Holm {s['holm']:.3f}" if name in CANDIDATES else ""
                cells.append("-" if s is None else f"{s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] ({s['better']:.0%}, p {s['p']:.3f}{h})")
            emit(f"| {name}{' (control)' if name in CONTROLS else ''} | " + " | ".join(cells) + " |")
        emit(f"\nn seeds per cell: {sorted({s['n'] for s in stats.values()})}")
    emit("\n#### Real minus control, log-loss gain (paired over seeds; positive = the graph / uncertainty signal adds over the control)\n")
    emit("| comparison | " + " | ".join(f"{s}-{c}" for s, c in BLOCKS) + " |"); emit("|---|" + "---|" * len(BLOCKS))
    label, lower = "log-loss", True; wides = {b: seed_gains(*b, "test_decisive_log_loss", True) for b in BLOCKS}
    for ctrl in CONTROLS:
        row = []
        for b in BLOCKS:
            w = wides[b]
            if "typed_decisive_coverage_unc" in w and ctrl in w:
                d = (w["typed_decisive_coverage_unc"] - w[ctrl]).dropna().to_numpy(); lo, hi = boot_ci(d); row.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (p {agg.wilcoxon_p(d):.3f})")
            else: row.append("-")
        emit(f"| coverage_unc minus {ctrl.split('coverage_unc_')[1]} | " + " | ".join(row) + " |")
    # verdict
    emit("\n#### Success criterion (log-loss Holm p < 0.05 in >= 2 of 4 blocks incl. both splits; mean AUC gain >= 0 in those blocks)\n")
    for name in CANDIDATES:
        ok = [b for b in BLOCKS if (name, *b) in results["log-loss"] and results["log-loss"][(name, *b)]["holm"] < 0.05 and results["log-loss"][(name, *b)]["mean"] > 0 and results["AUC"].get((name, *b), {"mean": -1})["mean"] >= 0]
        splits = {b[0] for b in ok}; passed = len(ok) >= 2 and splits == {"A", "B"}
        emit(f"- {name}: significant blocks {[f'{s}-{c}' for s, c in ok]} -> {'MET' if passed else 'NOT met'}")
    if out: open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

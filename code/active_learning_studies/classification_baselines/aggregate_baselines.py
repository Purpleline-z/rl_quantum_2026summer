#!/usr/bin/env python3
"""Summarise baseline metrics (mean, sd over splits; paired difference to a reference row) into a CSV and a markdown table.

  python3 aggregate_baselines.py results/frozen_paper_seeds [results/frozen_extra_seeds ...] --out results/baseline_table

Splits (seeds) are random re-partitions of the same 150 images, so their 28-image test sets overlap: the sd is split-to-split variability,
not a standard error of independent samples, and no significance test is attached.  The paired difference is per seed (same split).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = ("simclr_resnet18", "1nn")  # the baseline that already exists in the repository
COLUMNS = ["accuracy", "macro_f1", "HTR_recall", "HTR_precision"]


def summarise(metrics: pd.DataFrame, reference=REFERENCE) -> pd.DataFrame:
    ref = metrics[(metrics.extractor == reference[0]) & (metrics["head"] == reference[1])].set_index("seed").accuracy
    rows = []
    for (extractor, head), g in metrics.groupby(["extractor", "head"]):
        g = g.set_index("seed"); common = g.index.intersection(ref.index); diff = g.loc[common, "accuracy"] - ref.loc[common]
        row = {"extractor": extractor, "head": head, "n_splits": len(g), "n_test_per_split": int(g.n_test.iloc[0])}
        for c in COLUMNS: row[f"{c}_mean"], row[f"{c}_sd"] = g[c].mean(), g[c].std(ddof=1) if len(g) > 1 else float("nan")
        row["diff_vs_ref_mean"] = diff.mean(); row["splits_better_than_ref"] = int((diff > 0).sum()); row["splits_worse_than_ref"] = int((diff < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)


def markdown(table: pd.DataFrame, title: str, reference=REFERENCE) -> str:
    lines = [f"### {title}", "", f"Reference row for the paired difference: `{reference[0]}` + `{reference[1]}` (same split). 'better/worse' count splits.", "",
             "| extractor | head | splits | accuracy | macro-F1 | HTR recall | HTR precision | Δacc vs ref | better / worse |", "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.extractor} | {r.head} | {r.n_splits} | {r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f} | {r.macro_f1_mean:.3f} ± {r.macro_f1_sd:.3f} | "
                     f"{r.HTR_recall_mean:.2f} ± {r.HTR_recall_sd:.2f} | {r.HTR_precision_mean:.2f} ± {r.HTR_precision_sd:.2f} | {r.diff_vs_ref_mean:+.3f} | {r.splits_better_than_ref} / {r.splits_worse_than_ref} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("dirs", nargs="+", type=Path); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Frozen-feature baselines")
    args = parser.parse_args()
    metrics = pd.concat([pd.read_csv(d / "metrics.csv") for d in args.dirs], ignore_index=True)
    table = summarise(metrics); args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    unavailable = sorted({u["extractor"] for d in args.dirs if (d / "unavailable.json").exists() for u in pd.read_json(d / "unavailable.json").to_dict("records")})
    text = markdown(table, f"{args.title} ({table.n_splits.max()} splits, {int(table.n_test_per_split.iloc[0])} test images each)")
    if unavailable: text += "\nNot run (weights could not be downloaded in this environment): " + ", ".join(unavailable) + "\n"
    args.out.with_suffix(".md").write_text(text); print(text)


if __name__ == "__main__":
    main()

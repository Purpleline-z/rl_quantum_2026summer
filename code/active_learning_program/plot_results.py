"""Visualization for active learning experiment results.

Generates publication-quality figures for:
  - Learning curves (accuracy/Kendall-tau vs. labeled budget) per strategy
  - Bar charts comparing strategies at a fixed budget
  - Kendall-tau recovery curves from synthetic benchmark
  - Error bands from multiple seeds
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── color palette (colorblind-safe) ─────────────────────────────────────────
PALETTE = {
    "random":                        "#999999",
    "uncertainty":                   "#56B4E9",
    "cluster_quota_uncertainty":     "#CC79A7",
    "cluster_margin_pairwise":       "#009E73",
    "core_set":                      "#F0E442",
    "mc_dropout_mutual_information": "#0072B2",
    "mc_dropout_probability_variance":"#D55E00",
    "mc_dropout_reward_variance":    "#E69F00",
    # legacy names kept for backward compatibility with older result files
    "uncertainty_type_aware":        "#E69F00",
    "uncertainty_avg":               "#56B4E9",
    "cluster_margin":                "#009E73",
    "coreset":                       "#F0E442",
    "mc_bald_type_aware":            "#0072B2",
    "fisher_information":            "#D55E00",
}
MARKERS = {
    "random":                        "o",
    "uncertainty":                   "^",
    "cluster_quota_uncertainty":     "*",
    "cluster_margin_pairwise":       "D",
    "core_set":                      "v",
    "mc_dropout_mutual_information": "P",
    "mc_dropout_probability_variance":"X",
    "mc_dropout_reward_variance":    "s",
    # legacy
    "uncertainty_type_aware":        "s",
    "uncertainty_avg":               "^",
    "cluster_margin":                "D",
    "coreset":                       "v",
    "mc_bald_type_aware":            "P",
    "fisher_information":            "X",
}
LABELS = {
    "random":                        "Random",
    "uncertainty":                   "Uncertainty",
    "cluster_quota_uncertainty":     "Cluster-Quota Uncertainty [ours]",
    "cluster_margin_pairwise":       "Cluster-Margin",
    "core_set":                      "Core-Set",
    "mc_dropout_mutual_information": "MC-BALD [ours]",
    "mc_dropout_probability_variance":"MC Prob-Variance",
    "mc_dropout_reward_variance":    "MC Reward-Variance",
    # legacy
    "uncertainty_type_aware":        "Uncertainty (type-cond.) [ours]",
    "uncertainty_avg":               "Uncertainty (avg. heads)",
    "cluster_margin":                "Cluster-Margin",
    "coreset":                       "Core-Set",
    "mc_bald_type_aware":            "MC-BALD (type-cond.) [ours]",
    "fisher_information":            "Fisher Info (type-cond.) [ours]",
}


def _strategy_style(name: str) -> dict:
    color  = PALETTE.get(name, "#333333")
    marker = MARKERS.get(name, "o")
    label  = LABELS.get(name, name.replace("_", " ").title())
    return dict(color=color, marker=marker, label=label)


def plot_learning_curves(
    results: dict[str, list[dict[str, Any]]],
    metric: str = "holdout_accuracy",
    title: str = "Active Learning: Accuracy vs. Labeled Budget",
    ylabel: str = "Holdout Accuracy",
    output_path: str | Path = "learning_curves.pdf",
    figsize: tuple[float, float] = (7, 4.5),
) -> None:
    """Plot one curve per strategy (mean ± std over seeds).

    ``results`` maps strategy_name → list of records with keys
    ``budget``, ``mean_<metric>``, ``std_<metric>`` (or just
    ``metric`` for single-seed runs).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for strategy, records in sorted(results.items()):
        style = _strategy_style(strategy)
        budgets = [r["budget"] for r in records]
        mean_key = f"mean_{metric}"
        std_key  = f"std_{metric}"
        if mean_key in records[0]:
            means = np.array([r[mean_key] for r in records])
            stds  = np.array([r[std_key]  for r in records])
        else:
            means = np.array([r[metric] for r in records])
            stds  = np.zeros_like(means)
        ax.plot(budgets, means, marker=style["marker"],
                color=style["color"], label=style["label"], linewidth=1.8, markersize=5)
        if stds.max() > 0:
            ax.fill_between(budgets, means - stds, means + stds,
                            alpha=0.18, color=style["color"])
    ax.set_xlabel("Labeled pairs", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.35)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0] - 0.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_bar_comparison(
    results: dict[str, list[dict[str, Any]]],
    budget: int,
    metric: str = "holdout_accuracy",
    ylabel: str = "Holdout Accuracy",
    title: str | None = None,
    output_path: str | Path = "bar_comparison.pdf",
    figsize: tuple[float, float] = (6, 4),
) -> None:
    """Bar chart comparing strategies at a single budget point."""
    strategies, means, stds = [], [], []
    for strategy, records in sorted(results.items()):
        closest = min(records, key=lambda r: abs(r["budget"] - budget))
        strategies.append(strategy)
        mean_key = f"mean_{metric}"
        std_key  = f"std_{metric}"
        means.append(closest.get(mean_key, closest.get(metric, 0.0)))
        stds.append(closest.get(std_key, 0.0))
    sort_order = np.argsort(means)[::-1]
    strategies = [strategies[i] for i in sort_order]
    means      = [means[i]      for i in sort_order]
    stds       = [stds[i]       for i in sort_order]
    colors     = [PALETTE.get(s, "#333333") for s in strategies]
    labels     = [LABELS.get(s, s.replace("_", " ").title()) for s in strategies]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(strategies))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=4, linewidth=1.2, elinewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f"Strategy comparison at budget={budget:,}", fontsize=12)
    ax.grid(axis="y", alpha=0.35)
    ymin = max(0, min(means) - 0.08)
    ax.set_ylim(bottom=ymin)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_kendall_tau_curves(
    benchmark_results: dict[str, list[tuple[int, float, float]]],
    output_path: str | Path = "synthetic_kendall_tau.pdf",
    figsize: tuple[float, float] = (7, 4.5),
) -> None:
    """Kendall-tau recovery curves from synthetic benchmark.

    ``benchmark_results`` maps strategy_name → list of
    (budget, mean_kendall_tau, std_kendall_tau).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for strategy, triples in sorted(benchmark_results.items()):
        style = _strategy_style(strategy)
        budgets  = [t[0] for t in triples]
        means    = np.array([t[1] for t in triples])
        stds     = np.array([t[2] for t in triples])
        ax.plot(budgets, means, marker=style["marker"],
                color=style["color"], label=style["label"], linewidth=1.8, markersize=5)
        ax.fill_between(budgets, means - stds, means + stds,
                        alpha=0.18, color=style["color"])
    ax.set_xlabel("Labeled pairs", fontsize=12)
    ax.set_ylabel("Kendall-τ (higher = better)", fontsize=12)
    ax.set_title("Synthetic Benchmark: Reward Recovery", fontsize=13, pad=8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4, label="Oracle")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_type_conditioned_gain(
    results_by_type: dict[str, dict[str, list[dict]]],
    budget: int,
    metric: str = "holdout_accuracy",
    output_path: str | Path = "type_conditioned_gain.pdf",
    figsize: tuple[float, float] = (8, 4),
) -> None:
    """Per-type accuracy showing gain of type-conditioned over type-agnostic.

    ``results_by_type`` maps reconstruction_type → strategy_name → records.
    """
    rtype_names = list(results_by_type.keys())
    x = np.arange(len(rtype_names))
    width = 0.35
    tc_means, tc_stds, avg_means, avg_stds = [], [], [], []
    for rt in rtype_names:
        strat_results = results_by_type[rt]
        for strategy, means_list, stds_list in [
            ("uncertainty_type_aware", tc_means, tc_stds),
            ("uncertainty_avg",        avg_means, avg_stds),
        ]:
            if strategy in strat_results:
                records = strat_results[strategy]
                closest = min(records, key=lambda r: abs(r["budget"] - budget))
                means_list.append(closest.get(f"mean_{metric}", closest.get(metric, 0.0)))
                stds_list.append(closest.get(f"std_{metric}", 0.0))
            else:
                means_list.append(0.0)
                stds_list.append(0.0)
    fig, ax = plt.subplots(figsize=figsize)
    tc_color  = PALETTE["uncertainty_type_aware"]
    avg_color = PALETTE["uncertainty_avg"]
    ax.bar(x - width/2, tc_means,  width, label="Type-cond. [ours]", color=tc_color,  alpha=0.85)
    ax.bar(x + width/2, avg_means, width, label="Type-avg (baseline)", color=avg_color, alpha=0.85)
    ax.errorbar(x - width/2, tc_means,  yerr=tc_stds,  fmt="none", color="black", capsize=3)
    ax.errorbar(x + width/2, avg_means, yerr=avg_stds, fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(rtype_names, fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Per-type accuracy at budget={budget:,}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot active learning results")
    parser.add_argument("results_file", help="Path to results JSON or JSONL file")
    parser.add_argument("--metric", default="test_accuracy",
                        help="Metric column to plot (default: test_accuracy)")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--budget", type=int, default=None,
                        help="Fixed budget for bar chart; if not given, uses maximum")
    parser.add_argument("--outdir", default=".", help="Output directory for figures")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = Path(args.results_file)
    if p.suffix == ".jsonl":
        all_records = _load_jsonl(p)
    else:
        with open(p) as f:
            all_records = json.load(f)

    # Expect records to have "strategy" and "budget" keys
    if isinstance(all_records, dict):
        # Already keyed by strategy
        results = all_records
    else:
        results: dict[str, list[dict]] = {}
        for rec in all_records:
            key = rec.get("strategy", "unknown")
            results.setdefault(key, []).append(rec)

    budget = args.budget or max(r["budget"] for recs in results.values() for r in recs)
    ylabel = args.ylabel or args.metric.replace("_", " ").title()
    title  = args.title or f"Active Learning: {ylabel} vs. Budget"

    plot_learning_curves(
        results, metric=args.metric, title=title, ylabel=ylabel,
        output_path=outdir / "learning_curves.pdf",
    )
    plot_bar_comparison(
        results, budget=budget, metric=args.metric, ylabel=ylabel,
        output_path=outdir / "bar_comparison.pdf",
    )
    print("Done. Figures saved to", outdir)


if __name__ == "__main__":
    main()

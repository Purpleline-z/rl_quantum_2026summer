#!/usr/bin/env python3
"""Generate traceable figures for ACADEMIC_REPORT_DRAFT.md without inventing results."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "paper_assets"


def save_protocol_diagrams() -> None:
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off")
    boxes = [(0.02, "Pair groups\n(initial / candidate)"), (0.27, "BT reward model\n(training only)"),
             (0.52, "Utility validation\n(select epoch / settings)"), (0.77, "Outer ideal test\nreport once")]
    for x, label in boxes:
        ax.text(x, .5, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=.6", "fc": "#eaf2f8", "ec": "#2471a3"})
    for x in (.16, .41, .66): ax.annotate("", xy=(x + .07, .5), xytext=(x, .5), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(.645, .18, "No feedback from outer test to training, selection, or stopping.", ha="center", color="#922b21", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "pipeline_and_evaluation_protocol.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8)); ax.axis("off")
    ax.text(.17, .7, "Ideal-image pool", ha="center", va="center", fontsize=11, bbox={"boxstyle": "round,pad=.6", "fc": "#fef9e7", "ec": "#b7950b"})
    for x, label, color in ((.45, "Reference\n(train anchors", "#d5f5e3"), (.65, "Utility validation\nchoose epoch", "#d6eaf8"), (.85, "Outer test\nnever train", "#fadbd8")):
        ax.text(x, .7, label, ha="center", va="center", fontsize=10, bbox={"boxstyle": "round,pad=.5", "fc": color, "ec": "#555555"})
        ax.annotate("", xy=(x - .09, .7), xytext=(.28 if x == .45 else x - .18, .7), arrowprops={"arrowstyle": "->"})
    ax.text(.5, .32, "SHA-256 content identities are unique across all three partitions.\nAny outer-test identity is removed from pairwise/unlabeled trajectory images and Bad anchors.", ha="center", va="center", fontsize=10, color="#922b21")
    fig.tight_layout(); fig.savefig(OUT / "leakage_safe_split_protocol.svg"); plt.close(fig)


def save_historical_budget_curve() -> None:
    source = RESULTS / "selection_benchmark" / "stage1_selector_curves_none" / "aggregate" / "strategy_budget_summary.csv"
    if not source.exists():
        return
    table = pd.read_csv(source)
    metric = "post_test_accuracy_mean"
    strategy = next((c for c in ("strategy", "Strategy") if c in table), None)
    budget = next((c for c in ("budget", "acquisition_budget") if c in table), None)
    if not all((metric, strategy, budget)):
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, group in table.groupby(strategy):
        ax.plot(group[budget], group[metric], marker="o", label=name)
    ax.set(xlabel="Acquired pair-group budget", ylabel="Mean historical outer-test accuracy",
           title="Historical diagnostic selector curves (not final leakage-safe evidence)")
    ax.legend(fontsize=7); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(OUT / "historical_selector_budget_curve.png", dpi=180); plt.close(fig)


def save_symmetry_and_encoder_figures() -> None:
    symmetry_source = RESULTS / "selection_benchmark" / "stage2_symmetry_factorial" / "aggregate" / "strategy_budget_summary.csv"
    if symmetry_source.exists():
        table = pd.read_csv(symmetry_source)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for axis, mode in zip(axes, ("none", "left_half_mirror", "symmetric_average")):
            subset = table[table["symmetry_mode"] == mode]
            for strategy, group in subset.groupby("strategy"):
                axis.plot(group["budget"], group["post_test_accuracy_mean"], marker="o", label=strategy)
            axis.set_title(mode.replace("_", " ")); axis.set_xlabel("Budget"); axis.grid(alpha=.25)
        axes[0].set_ylabel("Mean historical outer-test accuracy")
        axes[-1].legend(fontsize=6, loc="best")
        fig.suptitle("Historical symmetry factorial (diagnostic only)"); fig.tight_layout()
        fig.savefig(OUT / "historical_symmetry_factorial.png", dpi=180); plt.close(fig)

    encoder_source = RESULTS / "protocol_diagnostics" / "encoder_initialization_screen" / "aggregate" / "encoder_utility_validation_summary.csv"
    if encoder_source.exists():
        table = pd.read_csv(encoder_source)
        pivot = table.pivot(index="selector", columns="encoder_initialization", values="mean")
        errors = table.pivot(index="selector", columns="encoder_initialization", values="std")
        fig, ax = plt.subplots(figsize=(6, 4))
        pivot.plot.bar(yerr=errors, capsize=3, ax=ax, rot=0)
        ax.set(ylabel="Mean utility-validation accuracy", xlabel="Selector", title="Historical encoder screen (diagnostic only)")
        ax.grid(axis="y", alpha=.25); fig.tight_layout()
        fig.savefig(OUT / "historical_encoder_screen.png", dpi=180); plt.close(fig)


def save_missing_epoch_evidence_notice() -> None:
    fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
    ax.text(.5, .60, "EPOCH-WISE CURVE NOT AVAILABLE FOR HISTORICAL RUNS", ha="center", va="center", weight="bold", color="#922b21", fontsize=13)
    ax.text(.5, .34, "The historical artifacts report endpoints at selected epoch counts, not\nvalidation accuracy after every epoch. Therefore they cannot justify stopping at epoch 10.", ha="center", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "missing_historical_epoch_curve.svg"); plt.close(fig)


if __name__ == "__main__":
    save_protocol_diagrams()
    save_historical_budget_curve()
    save_symmetry_and_encoder_figures()
    save_missing_epoch_evidence_notice()
    print(f"Wrote paper assets to {OUT}")

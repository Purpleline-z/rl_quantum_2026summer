#!/usr/bin/env python3
"""Create publication-ready figures from completed Drive Task 3 result JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_cells(path: Path) -> pd.DataFrame:
    rows = [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]
    if not rows: raise RuntimeError(f"No completed result JSONs under {path}")
    return pd.DataFrame(rows)


def save_all(fig, destination: Path, name: str) -> list[Path]:
    files = []
    for suffix in ("png", "pdf", "svg"):
        path = destination / f"{name}.{suffix}"; fig.savefig(path, dpi=300, bbox_inches="tight"); files.append(path)
    plt.close(fig); return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-output", type=Path, required=True); parser.add_argument("--wandb-project")
    args = parser.parse_args(); root = args.drive_output.expanduser().resolve(); out = root / "publication_ready_figures"; out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    calibration = pd.read_csv(root / "frozen_task3b_protocol" / "three_seed_validation_calibration_summary.csv")
    for budget, group in calibration.groupby("budget"):
        pivot = group.pivot(index="epochs", columns="learning_rate", values="mean_validation_accuracy")
        fig, ax = plt.subplots(figsize=(5.5, 3.8)); image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set(xticks=range(len(pivot.columns)), xticklabels=[f"{x:.0e}" for x in pivot.columns], yticks=range(len(pivot.index)), yticklabels=pivot.index, xlabel="Learning rate", ylabel="Maximum epochs", title=f"Validation-only schedule calibration | budget {budget} | SimCLR | n=3")
        fig.colorbar(image, ax=ax, label="Mean utility-validation accuracy"); produced += save_all(fig, out, f"validation_schedule_heatmap_budget_{budget}")
    final = load_cells(root / "task3c_final_strategy_cells")
    summary = final.groupby(["strategy", "budget"], as_index=False).agg(mean=("outer_test_accuracy", "mean"), sd=("outer_test_accuracy", "std"), n=("seed", "nunique"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for strategy, group in summary.groupby("strategy"):
        ax.errorbar(group.budget, group["mean"], yerr=group.sd.fillna(0), marker="o", capsize=3, label=strategy)
    ax.set(xlabel="Acquired pair-group budget", ylabel="Outer-test reconstruction accuracy", title="Identity-safe Task 3c | SimCLR | n=3 | validation-selected schedule")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); produced += save_all(fig, out, "eight_strategy_outer_test_accuracy_curve")
    random = final[final.strategy.eq("random")][["seed", "budget", "outer_test_accuracy"]].rename(columns={"outer_test_accuracy": "random_accuracy"})
    paired = final.merge(random, on=["seed", "budget"]); paired["difference_vs_random"] = paired.outer_test_accuracy - paired.random_accuracy
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for strategy, group in paired[~paired.strategy.eq("random")].groupby("strategy"):
        ax.plot(group.budget, group.groupby("budget").difference_vs_random.mean(), marker="o", label=strategy)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Budget", ylabel="Paired outer-test difference vs random", title="Per-seed paired strategy effect | SimCLR | n=3")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); produced += save_all(fig, out, "paired_outer_test_difference_vs_random")
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, group in final.groupby("strategy"):
        ax.scatter(group.budget, group.outer_test_accuracy, alpha=.7, label=strategy)
    ax.set(xlabel="Budget", ylabel="Outer-test accuracy", title="Seed-level outer-test outcomes | SimCLR | n=3")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); produced += save_all(fig, out, "seed_level_outer_test_scatter")
    audit_rows = [{"seed": row.seed, **row.identity_audit} for row in final.itertuples()]
    audit = pd.DataFrame(audit_rows).groupby("seed", as_index=False).first(); audit.to_csv(out / "identity_safe_audit_summary.csv", index=False)
    columns = [c for c in audit.columns if c.endswith("identity_overlap")]
    fig, ax = plt.subplots(figsize=(7, 3.8)); audit.set_index("seed")[columns].plot.bar(ax=ax)
    ax.set(ylabel="Overlap count (must be zero)", xlabel="Seed", title="SHA-256 identity-safe split audit"); ax.axhline(0, color="black", linewidth=.8)
    produced += save_all(fig, out, "identity_safe_split_audit")
    pd.DataFrame({"file": [str(path) for path in produced]}).to_csv(out / "figure_manifest.csv", index=False)
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, job_type="publication_figures", name="simclr-task3-publication-figures")
        artifact = wandb.Artifact("simclr-identity-safe-task3-figures", type="publication_figures"); artifact.add_dir(str(out)); run.log_artifact(artifact); run.finish()
    print(f"Wrote {len(produced)} publication files to {out}")


if __name__ == "__main__":
    main()

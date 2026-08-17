#!/usr/bin/env python3
"""Study-scoped runner for the result-oriented five-selector benchmark.

It never aggregates by globbing a shared result directory: every result must be
named by the immutable manifest supplied to the command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import measure_accuracy_at_each_new_label_budget as budget_runner
from project_file_locations import RESULT_ROOT


STUDIES = RESULT_ROOT / "selection_benchmark"
SEEDS = (42, 79, 123, 202, 303)
BUDGETS = (10, 25, 50, 75, 100)
SELECTORS = ("random", "uncertainty", "uncertainty_diversity", "cluster_quota_uncertainty", "core_set")
MC_SELECTORS = ("mc_dropout_probability_variance", "mc_dropout_mutual_information", "mc_dropout_reward_variance")
EXPLORATORY_SELECTORS = MC_SELECTORS + ("cluster_margin_pairwise",)
RUNNABLE_SELECTORS = SELECTORS + EXPLORATORY_SELECTORS
LEGACY_NAMES = {"cluster_diverse": "cluster_quota_uncertainty"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def study_root(study_id: str) -> Path:
    if not study_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Study ID may contain only letters, numbers, underscores, and hyphens.")
    return STUDIES / study_id


def job_id(spec: dict) -> str:
    return budget_runner.job_id(spec)


def create_manifest(study_id: str, strategies: tuple[str, ...], budgets: tuple[int, ...],
                    symmetry_modes: tuple[str, ...], seeds: tuple[int, ...] = SEEDS,
                    mc_dropout: bool = False, run_nmf_diagnostic: bool = True) -> Path:
    unknown = set(strategies) - set(RUNNABLE_SELECTORS)
    if unknown:
        raise ValueError(f"Unsupported benchmark selectors: {sorted(unknown)}")
    root = study_root(study_id)
    jobs = []
    for seed in seeds:
        for budget in budgets:
            for strategy in strategies:
                for symmetry_mode in symmetry_modes:
                    spec = {"study_id": study_id, "seed": seed, "budget": budget, "strategy": strategy,
                            "symmetry_mode": symmetry_mode, "epoch": 3, "lr": 1e-4,
                            "initial_pairs": 50, "candidate_pairs": 120,
                            "acquisition_mode": "single-shot", "diversity_lambda": .5,
                            "run_nmf_diagnostic": run_nmf_diagnostic}
                    if mc_dropout:
                        spec.update(dropout_p=.2, mc_samples=20)
                    spec["job_id"] = job_id(spec)
                    jobs.append(spec)
    manifest = {
        "study_id": study_id,
        "purpose": "Result-oriented active-pair selection benchmark",
        "protocol": {"seeds": seeds, "budgets": budgets, "selectors": strategies,
                     "symmetry_modes": symmetry_modes, "epoch": 3, "learning_rate": 1e-4,
                     "initial_pair_groups": 50, "candidate_pair_groups": 120,
                     "acquisition_mode": "single-shot", "uncertainty_diversity_lambda": .5,
                     "mc_dropout": {"enabled": mc_dropout, "dropout_p": .2 if mc_dropout else None,
                                    "mc_samples": 20 if mc_dropout else None},
                     "run_nmf_diagnostic": run_nmf_diagnostic},
        "jobs": jobs,
    }
    path = root / "study_manifest.json"
    atomic_json(manifest, path)
    return path


def manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("study_id") != path.parent.name:
        raise ValueError("Manifest study_id does not match its containing study directory.")
    return value


def run_job(manifest_path: Path, requested: str, data_root: str | None = None) -> None:
    value = manifest(manifest_path)
    matches = [job for job in value["jobs"] if job["job_id"] == requested or job["job_id"].startswith(requested)]
    if len(matches) != 1:
        raise ValueError(f"Expected one job for {requested!r}, found {len(matches)}.")
    original_out, original_data = budget_runner.OUT, budget_runner.DATA_ROOT
    try:
        budget_runner.OUT = manifest_path.parent
        budget_runner.DATA_ROOT = data_root
        budget_runner.run_job(matches[0])
    finally:
        budget_runner.OUT, budget_runner.DATA_ROOT = original_out, original_data


def _row_from_result(path: Path, provenance: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    spec = result["spec"]
    return {"seed": spec["seed"], "budget": spec["budget"], "strategy": spec["strategy"],
            "symmetry_mode": spec["symmetry_mode"], "post_test_accuracy": result["post"]["test_accuracy"],
            "batch_utility": result["batch_utility"], "pre_test_accuracy": result["pre"]["test_accuracy"],
            "job_id": spec["job_id"], "provenance": provenance, "source_path": str(path),
            "source_sha256": sha256(path)}


def _validate_grid(frame: pd.DataFrame, strategies: tuple[str, ...], budgets: tuple[int, ...], modes: tuple[str, ...]) -> None:
    expected = {(seed, budget, strategy, mode) for seed in SEEDS for budget in budgets for strategy in strategies for mode in modes}
    observed = set(frame[["seed", "budget", "strategy", "symmetry_mode"]].itertuples(index=False, name=None))
    if observed != expected:
        missing, extra = expected - observed, observed - expected
        raise ValueError(f"Benchmark grid mismatch. Missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}")


def normalize_historical_endpoint(output: Path) -> Path:
    """Freeze the compatible five-seed, budget-100 evidence with provenance."""
    local_source = RESULT_ROOT / "strategy_followup_analysis" / "five_seed_extension" / "five_seed_per_run.csv"
    curated_source = RESULT_ROOT / "selected_summaries" / "endpoint_extension_5seed" / "tables" / "per_seed_nonprimary_outcomes.csv"
    source = local_source if local_source.exists() else curated_source
    if not source.exists():
        raise FileNotFoundError(f"Neither local nor Git-tracked endpoint evidence exists: {local_source} or {curated_source}")
    frame = pd.read_csv(source)
    frame["strategy"] = frame.strategy.replace(LEGACY_NAMES)
    wanted = frame[(frame.seed.isin(SEEDS)) & (frame.budget == 100) &
                   (frame.strategy.isin(("uncertainty_diversity", "cluster_quota_uncertainty", "core_set")))].copy()
    expected = {(seed, strategy) for seed in SEEDS for strategy in ("uncertainty_diversity", "cluster_quota_uncertainty", "core_set")}
    found = set(wanted[["seed", "strategy"]].itertuples(index=False, name=None))
    if found != expected:
        raise ValueError(f"Historical endpoint lacks expected seed/strategy cells: {sorted(expected - found)}")
    if "epoch" in wanted and (wanted.epoch.nunique() != 1 or int(wanted.epoch.iloc[0]) != 3):
        raise ValueError("Historical endpoint is not compatible with epoch-3, budget-100 protocol.")
    if "selected_pairs" in wanted and (wanted.selected_pairs.nunique() != 1 or int(wanted.selected_pairs.iloc[0]) != 100):
        raise ValueError("Historical endpoint is not compatible with budget-100 protocol.")
    columns = ["seed", "budget", "strategy", "pre_test_accuracy", "post_test_accuracy", "batch_utility", "job_id"]
    normalized = wanted[columns].copy()
    normalized["source_result"] = wanted.get("source_result", wanted.get("source_family", "Git-tracked curated endpoint evidence"))
    normalized["symmetry_mode"] = "none"
    normalized["provenance"] = "reused_historical_budget100_endpoint"
    normalized["source_csv"] = str(source)
    normalized["source_csv_sha256"] = sha256(source)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "shared_endpoint_5seed_normalized.csv"
    normalized.to_csv(path, index=False)
    atomic_json({"source": str(source), "source_sha256": sha256(source), "legacy_strategy_mapping": LEGACY_NAMES,
                 "compatibility": {"seeds": SEEDS, "budget": 100, "epoch": 3, "selected_pair_groups": 100,
                                   "symmetry_mode": "none", "status": "compatible"}},
                output / "shared_endpoint_5seed_provenance.json")
    return path


def normalize_existing_primary_curve(output: Path) -> Path:
    source_root = RESULT_ROOT / "budget_curve_study" / "jobs"
    rows = []
    missing_local = False
    for seed in SEEDS:
        for budget in BUDGETS:
            for strategy in ("random", "uncertainty"):
                identifier = job_id({"seed": seed, "budget": budget, "strategy": strategy, "symmetry_mode": "none"})
                result = source_root / identifier / "result.json"
                if not result.exists():
                    missing_local = True
                    break
                rows.append(_row_from_result(result, "reused_existing_primary_curve"))
            if missing_local:
                break
        if missing_local:
            break
    if missing_local:
        source = RESULT_ROOT / "selected_summaries" / "budget_curve_5seed" / "tables" / "per_seed_performance_by_acquisition_budget.csv"
        if not source.exists():
            raise FileNotFoundError(f"Git-tracked primary-curve evidence missing: {source}")
        curated = pd.read_csv(source)
        frame = curated[(curated.seed.isin(SEEDS)) & (curated.budget.isin(BUDGETS)) &
                        (curated.strategy.isin(("random", "uncertainty"))) & curated.symmetry_mode.eq("none")].copy()
        frame["provenance"] = "reused_git_tracked_primary_curve"
        frame["source_path"] = str(source)
        frame["source_sha256"] = sha256(source)
    else:
        frame = pd.DataFrame(rows)
    _validate_grid(frame, ("random", "uncertainty"), BUDGETS, ("none",))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "shared_primary_curve_normalized.csv"
    frame.to_csv(path, index=False)
    return path


def prepare_reused_evidence(manifest_path: Path) -> Path:
    """Materialize auditable source snapshots before any new job is run."""
    root = manifest(manifest_path)
    if root["study_id"] != "stage1_selector_curves_none":
        raise ValueError("Reused endpoint evidence is defined only for the Stage 1 none-mode selector benchmark.")
    output = manifest_path.parent / "reused_evidence"
    normalize_existing_primary_curve(output)
    normalize_historical_endpoint(output)
    return output


def collect_new_jobs(manifest_path: Path) -> pd.DataFrame:
    value = manifest(manifest_path)
    rows, missing = [], []
    for spec in value["jobs"]:
        result = manifest_path.parent / "jobs" / spec["job_id"] / "result.json"
        if not result.exists():
            missing.append(spec["job_id"])
        else:
            rows.append(_row_from_result(result, "new_benchmark_run"))
    if missing:
        raise ValueError("Study is incomplete; do not aggregate partial evidence. Missing jobs:\n" + "\n".join(missing))
    return pd.DataFrame(rows)


def run_manifest(manifest_path: Path, data_root: str | None = None) -> None:
    """Run only the immutable manifest's incomplete jobs; safe after interruption."""
    value = manifest(manifest_path)
    completed = 0
    for spec in value["jobs"]:
        result = manifest_path.parent / "jobs" / spec["job_id"] / "result.json"
        if result.exists():
            print(f"{spec['job_id']}: completed")
            completed += 1
            continue
        run_job(manifest_path, spec["job_id"], data_root)
    print(f"{value['study_id']}: {completed} previously complete; {len(value['jobs'])} total jobs.")


def _write_summary(frame: pd.DataFrame, output: Path, title_prefix: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_outcomes.csv", index=False)
    summary = frame.groupby(["budget", "strategy", "symmetry_mode"], as_index=False).agg(
        n=("seed", "count"), post_test_accuracy_mean=("post_test_accuracy", "mean"),
        post_test_accuracy_std=("post_test_accuracy", "std"), batch_utility_mean=("batch_utility", "mean"),
        batch_utility_std=("batch_utility", "std"))
    summary.to_csv(output / "strategy_budget_summary.csv", index=False)
    random = frame[frame.strategy == "random"][["seed", "budget", "symmetry_mode", "post_test_accuracy", "batch_utility"]]
    paired = frame.merge(random, on=["seed", "budget", "symmetry_mode"], suffixes=("", "_random"))
    if not paired.empty:
        paired["post_test_accuracy_difference_vs_random"] = paired.post_test_accuracy - paired.post_test_accuracy_random
        paired["batch_utility_difference_vs_random"] = paired.batch_utility - paired.batch_utility_random
    paired.to_csv(output / "paired_differences_vs_random.csv", index=False)
    for metric, label, filename in (("post_test_accuracy", "Post-acquisition fixed ideal-test accuracy", "post_test_accuracy_by_budget.png"),
                                   ("batch_utility", "Batch utility", "batch_utility_by_budget.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for (strategy, mode), group in frame.groupby(["strategy", "symmetry_mode"]):
            for _, seed_rows in group.groupby("seed"):
                ax.plot(seed_rows.budget, seed_rows[metric], color="0.8", alpha=.45, linewidth=.6)
            stats = group.groupby("budget")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats.budget, stats["mean"], yerr=stats["std"], marker="o", capsize=3, label=f"{strategy} | {mode}")
        ax.set(title=f"{title_prefix}: {label}", xlabel="Acquired pair groups", ylabel=label)
        ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)


def _write_baseline_comparison(frame: pd.DataFrame, output: Path, baseline_strategies: tuple[str, ...]) -> None:
    """Compare only matching seed/budget cells and retain the source provenance."""
    stage1 = STUDIES / "stage1_selector_curves_none" / "aggregate" / "per_seed_outcomes.csv"
    if not stage1.exists():
        atomic_json({"comparison_available": False, "reason": "Stage 1 aggregate is not available yet; no winner is claimed."},
                    output / "baseline_comparison_status.json")
        return
    baseline = pd.read_csv(stage1)
    baseline = baseline[baseline.strategy.isin(baseline_strategies) & baseline.symmetry_mode.eq("none")]
    candidate = frame[["seed", "budget", "strategy", "post_test_accuracy", "batch_utility"]]
    joined = candidate.merge(baseline[["seed", "budget", "strategy", "post_test_accuracy", "batch_utility", "source_path", "source_sha256"]],
                             on=["seed", "budget"], suffixes=("", "_baseline"))
    joined = joined[joined.strategy_baseline.isin(baseline_strategies)]
    joined["post_test_accuracy_difference"] = joined.post_test_accuracy - joined.post_test_accuracy_baseline
    joined["batch_utility_difference"] = joined.batch_utility - joined.batch_utility_baseline
    joined.to_csv(output / "paired_differences_vs_stage1_baselines.csv", index=False)
    atomic_json({"comparison_available": True, "source": str(stage1), "source_sha256": sha256(stage1),
                 "baseline_strategies": baseline_strategies, "interpretation": "paired descriptive comparison only"},
                output / "baseline_comparison_status.json")


def aggregate_stage1(manifest_path: Path) -> Path:
    root = manifest_path.parent
    new = collect_new_jobs(manifest_path)
    if set(new.strategy) != {"uncertainty_diversity", "cluster_quota_uncertainty", "core_set"} or set(new.budget) != {10, 25, 50, 75} or set(new.symmetry_mode) != {"none"}:
        raise ValueError("Stage 1 manifest must contain only the 60 missing none-mode selector cells.")
    reused = prepare_reused_evidence(manifest_path)
    primary = pd.read_csv(reused / "shared_primary_curve_normalized.csv")
    endpoint = pd.read_csv(reused / "shared_endpoint_5seed_normalized.csv")
    endpoint = endpoint.rename(columns={"source_result": "source_path", "source_csv_sha256": "source_sha256"})
    combined = pd.concat([primary, endpoint, new], ignore_index=True, sort=False)
    _validate_grid(combined, SELECTORS, BUDGETS, ("none",))
    output = root / "aggregate"
    _write_summary(combined, output, "Five-selector benchmark (none)")
    atomic_json({"manifest": str(manifest_path), "new_jobs": len(new), "reused_rows": len(primary) + len(endpoint),
                 "completed_grid_cells": len(combined), "status": "completed"}, output / "aggregation_manifest.json")
    return output


def aggregate_stage2(manifest_path: Path, stage1_aggregate: Path) -> Path:
    """Join completed none-mode evidence to the two new symmetry modes only."""
    new = collect_new_jobs(manifest_path)
    if set(new.strategy) != set(SELECTORS) or set(new.budget) != set(BUDGETS) or set(new.symmetry_mode) != {"left_half_mirror", "symmetric_average"}:
        raise ValueError("Stage 2 manifest must contain exactly the two non-baseline symmetry modes for the full five-selector grid.")
    baseline = pd.read_csv(stage1_aggregate / "per_seed_outcomes.csv")
    _validate_grid(baseline, SELECTORS, BUDGETS, ("none",))
    combined = pd.concat([baseline, new], ignore_index=True, sort=False)
    _validate_grid(combined, SELECTORS, BUDGETS, ("none", "left_half_mirror", "symmetric_average"))
    output = manifest_path.parent / "aggregate"
    _write_summary(combined, output, "Five-selector × symmetry factorial")
    interaction = combined.groupby(["strategy", "symmetry_mode", "budget"], as_index=False).agg(
        n=("seed", "count"), post_test_accuracy_mean=("post_test_accuracy", "mean"),
        post_test_accuracy_std=("post_test_accuracy", "std"), batch_utility_mean=("batch_utility", "mean"),
        batch_utility_std=("batch_utility", "std"))
    interaction.to_csv(output / "selector_symmetry_budget_interaction.csv", index=False)
    atomic_json({"manifest": str(manifest_path), "baseline_aggregate": str(stage1_aggregate), "new_jobs": len(new),
                 "completed_grid_cells": len(combined), "status": "completed"}, output / "aggregation_manifest.json")
    return output


def aggregate_mc_screen(manifest_path: Path) -> Path:
    frame = collect_new_jobs(manifest_path)
    if set(frame.strategy) != set(MC_SELECTORS) or set(frame.budget) != {10, 25} or set(frame.symmetry_mode) != {"none"}:
        raise ValueError("MC screen manifest does not match the defined 30-job low-budget screening protocol.")
    _validate_grid(frame, MC_SELECTORS, (10, 25), ("none",))
    output = manifest_path.parent / "aggregate"
    _write_summary(frame, output, "MC-dropout low-budget screening")
    _write_baseline_comparison(frame, output, ("uncertainty", "uncertainty_diversity"))
    atomic_json({"manifest": str(manifest_path), "new_jobs": len(frame), "completed_grid_cells": len(frame),
                 "status": "completed", "winner_claim": "not made by this screening"}, output / "aggregation_manifest.json")
    return output


def aggregate_cluster_margin(manifest_path: Path) -> Path:
    frame = collect_new_jobs(manifest_path)
    if set(frame.strategy) != {"cluster_margin_pairwise"} or set(frame.budget) != set(BUDGETS) or set(frame.symmetry_mode) != {"none"}:
        raise ValueError("Cluster-Margin manifest does not match the defined 25-job full-budget protocol.")
    _validate_grid(frame, ("cluster_margin_pairwise",), BUDGETS, ("none",))
    output = manifest_path.parent / "aggregate"
    _write_summary(frame, output, "Cluster-Margin pairwise curve")
    _write_baseline_comparison(frame, output, ("random", "uncertainty", "uncertainty_diversity"))
    atomic_json({"manifest": str(manifest_path), "new_jobs": len(frame), "completed_grid_cells": len(frame),
                 "status": "completed", "baseline_join": "only if Stage 1 aggregate exists, with source hash"}, output / "aggregation_manifest.json")
    return output


def generate_stage1() -> Path:
    return create_manifest("stage1_selector_curves_none", ("uncertainty_diversity", "cluster_quota_uncertainty", "core_set"),
                           (10, 25, 50, 75), ("none",))


def generate_stage2() -> Path:
    return create_manifest("stage2_symmetry_factorial", SELECTORS, BUDGETS, ("left_half_mirror", "symmetric_average"))


def generate_mc_screen() -> Path:
    return create_manifest("mc_dropout_screen_low_budget", MC_SELECTORS, (10, 25), ("none",),
                           mc_dropout=True, run_nmf_diagnostic=False)


def generate_cluster_margin() -> Path:
    return create_manifest("cluster_margin_curve_none", ("cluster_margin_pairwise",), BUDGETS, ("none",),
                           run_nmf_diagnostic=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-stage1-manifest")
    sub.add_parser("generate-stage2-manifest")
    sub.add_parser("generate-mc-screen-manifest")
    sub.add_parser("generate-cluster-margin-manifest")
    run = sub.add_parser("run-job"); run.add_argument("--manifest", type=Path, required=True); run.add_argument("--job-id", required=True); run.add_argument("--data-root")
    run_all = sub.add_parser("run-manifest"); run_all.add_argument("--manifest", type=Path, required=True); run_all.add_argument("--data-root")
    reused = sub.add_parser("prepare-reused-evidence"); reused.add_argument("--manifest", type=Path, required=True)
    aggregate = sub.add_parser("aggregate-stage1"); aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate2 = sub.add_parser("aggregate-stage2"); aggregate2.add_argument("--manifest", type=Path, required=True); aggregate2.add_argument("--stage1-aggregate", type=Path, required=True)
    aggregate_mc = sub.add_parser("aggregate-mc-screen"); aggregate_mc.add_argument("--manifest", type=Path, required=True)
    aggregate_cm = sub.add_parser("aggregate-cluster-margin"); aggregate_cm.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-stage1-manifest": print(generate_stage1())
    elif args.command == "generate-stage2-manifest": print(generate_stage2())
    elif args.command == "generate-mc-screen-manifest": print(generate_mc_screen())
    elif args.command == "generate-cluster-margin-manifest": print(generate_cluster_margin())
    elif args.command == "run-job": run_job(args.manifest, args.job_id, args.data_root)
    elif args.command == "run-manifest": run_manifest(args.manifest, args.data_root)
    elif args.command == "prepare-reused-evidence": print(prepare_reused_evidence(args.manifest))
    elif args.command == "aggregate-stage1": print(aggregate_stage1(args.manifest))
    elif args.command == "aggregate-stage2": print(aggregate_stage2(args.manifest, args.stage1_aggregate))
    elif args.command == "aggregate-mc-screen": print(aggregate_mc_screen(args.manifest))
    else: print(aggregate_cluster_margin(args.manifest))


if __name__ == "__main__":
    main()

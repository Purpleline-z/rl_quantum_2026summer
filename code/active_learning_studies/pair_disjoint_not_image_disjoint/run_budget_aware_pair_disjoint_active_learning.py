#!/usr/bin/env python3
"""Budget-aware, Drive-checkpointed pair-disjoint active-learning experiments.

Calibration uses a deterministic random reference acquisition at every budget
and validation accuracy only.  It never chooses one epoch count for all data
budgets.  Final strategy cells use the validation-selected setting for their
own encoder and budget, then report outer-test accuracy only after training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT / "active_learning_program"))

from pairwise_active_learning_pipeline import Config, Experiment
from resumable_model_training import train_with_epoch_checkpoints


APPROVED_STRATEGIES = (
    "random",
    "uncertainty",
    "core_set",
    "mc_dropout_mutual_information",
    "mc_dropout_probability_variance",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def task_id(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def make_experiment(protocol: dict[str, Any], output: Path, spec: dict[str, Any], epochs: int, device: str) -> Experiment:
    cfg = Config(
        initial_pairs=int(protocol["initial_pair_groups"]),
        candidate_pairs=int(protocol["maximum_acquired_pair_groups"]),
        epochs=int(epochs), train_batch_size=int(protocol["train_batch_size"]),
        lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]),
        seed=int(spec["seed"]), device=device, dropout_p=float(protocol["dropout_probability"]),
        encoder_initialization=str(spec["encoder"]), acquisition_mode="single-shot",
        dataset_version=str(protocol["dataset_version"]),
        manifest_dir=str(output / "pair_disjoint_audits" / task_id(spec)),
    )
    value = Experiment(cfg)
    value.output = output
    value.utility_cache_path = output / "utility_cache_unused.json"
    value.utility_cache = {}
    return value


def validation_calibration_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": seed, "encoder": encoder, "budget": budget, "learning_rate": lr,
         "weight_decay": protocol["default_weight_decay"], "epochs": epochs}
        for seed in protocol["seeds"] for encoder in protocol["encoder_initializations"]
        for budget in protocol["budgets"] for lr in protocol["learning_rates"]
        for epochs in protocol["epochs"]
    ]


def deterministic_random_reference_ids(candidates: list[str], seed: int, budget: int) -> list[str]:
    """Choose calibration labels without inspecting any candidate preference label."""
    return random.Random(seed * 1_000_003 + budget).sample(candidates, min(budget, len(candidates)))


def completed_in_drive_or_git(output: Path, drive_directory: str, git_directory_prefix: str, identifier: str) -> bool:
    """Treat pushed result JSON as completed when a queue migrates to a new account."""
    filename = f"{identifier}.json"
    if (output / drive_directory / filename).exists():
        return True
    return any((HERE / "results").glob(f"{git_directory_prefix}_account_*/{filename}"))


def run_validation_calibration_cell(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str,
                                    completed_results_directory: Path | None = None) -> None:
    identifier = task_id(task)
    result_root = completed_results_directory or output / "budget_aware_validation_calibration"
    result = result_root / f"{identifier}.json"
    if result.exists() or completed_in_drive_or_git(output, "budget_aware_validation_calibration", "budget_aware_validation_calibration", identifier):
        print(f"completed budget-aware calibration cell: {result.name}", flush=True)
        return
    run_output = output / "budget_aware_validation_runs" / identifier
    value = make_experiment(protocol, run_output, task, int(task["epochs"]), device)
    initial, candidates = value.load_and_split()
    selected = deterministic_random_reference_ids(candidates, int(task["seed"]), int(task["budget"]))
    pair_ids = initial + selected
    checkpoint = output / "resumable_checkpoints" / "budget_aware_validation_calibration" / f"{identifier}.pth"
    phase = f"budget-aware-validation-{identifier}"
    model, metrics, paused = train_with_epoch_checkpoints(
        value, pair_ids, checkpoint, phase, None,
        heartbeat_seconds=int(protocol["checkpoint_heartbeat_seconds"]), checkpoint_enabled=True,
    )
    if paused or model is None:
        raise RuntimeError("Calibration paused; rerun the identical command.")
    validation = value.evaluate(model, "utility_validation")
    write_json({**task, "initial_pair_groups": len(initial), "reference_selected_pair_ids": selected,
                "training_pairwise_accuracy": metrics["pairwise_accuracy"],
                "optimizer_updates": metrics.get("optimizer_updates"),
                "utility_validation_accuracy": validation["test_accuracy"],
                "utility_validation_by_class": validation["by_class"],
                "outer_test_not_evaluated": True, "checkpoint": str(checkpoint)}, result)
    # A completed JSON is the durable research record.  The checkpoint is only
    # needed for an interrupted cell and would otherwise grow Drive without
    # bound across the 600-cell grid.
    checkpoint.unlink(missing_ok=True)
    print(f"saved validation-only calibration cell: {result}", flush=True)


def run_bounded_validation_calibration_queue(protocol: dict[str, Any], output: Path, account_index: int, account_count: int,
                                             maximum_cells: int, device: str,
                                             completed_results_directory: Path | None = None) -> None:
    assigned = [task for index, task in enumerate(validation_calibration_tasks(protocol)) if index % account_count == account_index]
    pending = [task for task in assigned if not completed_in_drive_or_git(output, "budget_aware_validation_calibration", "budget_aware_validation_calibration", task_id(task))]
    scheduled = pending if maximum_cells == 0 else pending[:maximum_cells]
    scope = "until complete" if maximum_cells == 0 else f"at most {maximum_cells} cells"
    print(f"account {account_index + 1}/{account_count}: {len(assigned) - len(pending)}/{len(assigned)} calibration cells complete; running {scope}", flush=True)
    for number, task in enumerate(scheduled, start=1):
        run_validation_calibration_cell(protocol, output, task, device, completed_results_directory)
        print(f"calibration progress {number}/{len(scheduled)}; account remaining={len(pending) - number}", flush=True)


def aggregate_budget_aware_validation_calibration(protocol: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for directory in sorted((HERE / "results").glob("budget_aware_validation_calibration_account_*")):
        for path in directory.glob("*.json"):
            row = read_json(path); row["source_directory"] = directory.name; rows.append(row)
    expected = pd.DataFrame(validation_calibration_tasks(protocol))
    keys = ["seed", "encoder", "budget", "learning_rate", "weight_decay", "epochs"]
    observed = pd.DataFrame(rows)
    if observed.empty:
        raise RuntimeError("No pushed budget-aware calibration JSON files were found.")
    duplicate = observed.duplicated(keys, keep=False)
    if duplicate.any():
        raise RuntimeError("Duplicate calibration task specifications were pushed; resolve them before aggregation.")
    matched = observed.merge(expected[keys], how="inner", on=keys)
    if len(matched) != len(expected):
        raise RuntimeError(f"Budget-aware calibration incomplete: found {len(matched)} expected cells, need {len(expected)}.")
    summary = matched.groupby(["encoder", "budget", "learning_rate", "weight_decay", "epochs"], as_index=False).agg(
        utility_validation_accuracy_mean=("utility_validation_accuracy", "mean"),
        utility_validation_accuracy_sd=("utility_validation_accuracy", "std"), seeds=("seed", "nunique"),
    )
    locked: dict[str, dict[str, dict[str, float | int]]] = {}
    for (encoder, budget), group in summary.groupby(["encoder", "budget"]):
        best = group.sort_values(["utility_validation_accuracy_mean", "epochs", "learning_rate"], ascending=[False, True, True]).iloc[0]
        locked.setdefault(str(encoder), {})[str(int(budget))] = {
            "learning_rate": float(best.learning_rate), "weight_decay": float(best.weight_decay), "epochs": int(best.epochs),
        }
    destination = HERE / "results" / "budget_aware_protocol"
    destination.mkdir(parents=True, exist_ok=True)
    matched.to_csv(destination / "per_seed_validation_calibration.csv", index=False)
    summary.to_csv(destination / "validation_calibration_summary.csv", index=False)
    write_json({"selection_endpoint": "utility_validation_accuracy_mean", "outer_test_used_for_selection": False,
                "protocol_by_encoder_and_acquisition_budget": locked,
                "rule": "The selected setting is shared by all acquisition strategies at its encoder and budget."},
               destination / "budget_aware_protocol_by_encoder_and_budget.json")
    print(f"Wrote budget-aware protocol from {len(matched)} validation-only cells to {destination}", flush=True)


def final_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    path = HERE / "results" / "budget_aware_protocol" / "budget_aware_protocol_by_encoder_and_budget.json"
    if not path.exists():
        raise RuntimeError("The budget-aware protocol is not Git-tracked yet; complete and aggregate validation calibration first.")
    selected = read_json(path)["protocol_by_encoder_and_acquisition_budget"]
    return [
        {"seed": seed, "encoder": encoder, "strategy": strategy, "budget": budget,
         "training_control": control,
         **({"fixed_optimizer_updates": int(protocol["fixed_total_optimizer_updates"])} if control == "fixed_total_optimizer_updates" else {}),
         **selected[encoder][str(budget)]}
        for seed in protocol["seeds"] for encoder in protocol["encoder_initializations"]
        for strategy in APPROVED_STRATEGIES for budget in protocol["budgets"]
        for control in ("budget_specific_epochs", "fixed_total_optimizer_updates")
    ]


def run_final_strategy_cell(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str) -> None:
    identifier = task_id(task)
    result = output / "budget_aware_final_strategy_cells" / f"{identifier}.json"
    if completed_in_drive_or_git(output, "budget_aware_final_strategy_cells", "budget_aware_final_strategy_cells", identifier):
        print(f"completed final strategy cell: {result.name}", flush=True)
        return
    run_output = output / "budget_aware_final_strategy_runs" / identifier
    value = make_experiment(protocol, run_output, task, int(task["epochs"]), device)
    initial, candidates = value.load_and_split()
    if task["training_control"] == "fixed_total_optimizer_updates":
        baseline, baseline_metrics = value.train(initial, max_optimizer_updates=int(task["fixed_optimizer_updates"]))
        baseline_checkpoint = None
    else:
        baseline_checkpoint = output / "resumable_checkpoints" / "budget_aware_final_baselines" / f"{identifier}.pth"
        baseline, baseline_metrics, paused = train_with_epoch_checkpoints(value, initial, baseline_checkpoint, f"budget-aware-baseline-{identifier}", None, heartbeat_seconds=int(protocol["checkpoint_heartbeat_seconds"]), checkpoint_enabled=True)
        if paused or baseline is None: raise RuntimeError("Baseline paused; rerun the identical command.")
    candidate_rows, cache = value.candidates_with_clusters(candidates, baseline)
    selected, _, _ = value.select(task["strategy"], candidate_rows, baseline, cache, [], budget=int(task["budget"]), labeled_ids=initial)
    selected_ids = [row["pair_id"] for row in selected]
    if task["training_control"] == "fixed_total_optimizer_updates":
        final_model, metrics = value.train(initial + selected_ids, max_optimizer_updates=int(task["fixed_optimizer_updates"]))
        final_checkpoint = None
    else:
        final_checkpoint = output / "resumable_checkpoints" / "budget_aware_final_models" / f"{identifier}.pth"
        final_model, metrics, paused = train_with_epoch_checkpoints(value, initial + selected_ids, final_checkpoint, f"budget-aware-final-{identifier}", None, heartbeat_seconds=int(protocol["checkpoint_heartbeat_seconds"]), checkpoint_enabled=True)
        if paused or final_model is None: raise RuntimeError("Final training paused; rerun the identical command.")
    validation, outer = value.evaluate(final_model, "utility_validation"), value.evaluate(final_model, "outer_test")
    write_json({**task, "initial_pair_groups": len(initial), "selected_pair_ids": selected_ids,
                "training_pairwise_accuracy": metrics["pairwise_accuracy"], "utility_validation_accuracy": validation["test_accuracy"],
                "outer_test_accuracy": outer["test_accuracy"], "outer_test_by_class": outer["by_class"],
                "baseline_optimizer_updates": baseline_metrics["optimizer_updates"],
                "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
                "final_checkpoint": str(final_checkpoint) if final_checkpoint else None}, result)
    # Keep resumable state only while a cell is incomplete.  Its compact result
    # JSON has already been atomically written before these checkpoint removals.
    if baseline_checkpoint: baseline_checkpoint.unlink(missing_ok=True)
    if final_checkpoint: final_checkpoint.unlink(missing_ok=True)
    print(f"saved final strategy cell: {result}", flush=True)


def run_bounded_final_strategy_queue(protocol: dict[str, Any], output: Path, account_index: int, account_count: int, maximum_cells: int, device: str) -> None:
    tasks = final_tasks(protocol)
    assigned = [task for index, task in enumerate(tasks) if index % account_count == account_index]
    pending = [task for task in assigned if not completed_in_drive_or_git(output, "budget_aware_final_strategy_cells", "budget_aware_final_strategy_cells", task_id(task))]
    scheduled = pending if maximum_cells == 0 else pending[:maximum_cells]
    scope = "until complete" if maximum_cells == 0 else f"at most {maximum_cells} cells"
    print(f"account {account_index + 1}/{account_count}: {len(assigned) - len(pending)}/{len(assigned)} final cells complete; running {scope}", flush=True)
    for number, task in enumerate(scheduled, start=1):
        run_final_strategy_cell(protocol, output, task, device)
        print(f"final progress {number}/{len(scheduled)}; account remaining={len(pending) - number}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run_bounded_validation_calibration_queue", "aggregate_budget_aware_validation_calibration", "run_bounded_final_strategy_queue"))
    parser.add_argument("--config", type=Path, default=HERE / "literature_backed_pair_disjoint_budget_curve_settings.json")
    parser.add_argument("--drive-output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--account-index", type=int)
    parser.add_argument("--account-count", type=int, default=4)
    parser.add_argument("--maximum-cells", type=int, default=0,
                        help="0 runs all remaining assigned cells automatically; a positive value is a debugging limit.")
    parser.add_argument("--completed-results-directory", type=Path,
                        help="Optional tracked directory for completed validation JSONs; useful when checkpoints are local rather than on Drive.")
    args = parser.parse_args(); protocol = read_json(args.config)
    if args.action == "aggregate_budget_aware_validation_calibration":
        aggregate_budget_aware_validation_calibration(protocol); return
    if args.drive_output is None or args.account_index is None:
        parser.error("queue actions require --drive-output and --account-index")
    if not 0 <= args.account_index < args.account_count: parser.error("--account-index must be in [0, account-count)")
    if args.maximum_cells < 0: parser.error("--maximum-cells must be non-negative")
    output = args.drive_output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.action == "run_bounded_validation_calibration_queue":
        completed_directory = (args.completed_results_directory.expanduser().resolve()
                               if args.completed_results_directory else None)
        if completed_directory:
            completed_directory.mkdir(parents=True, exist_ok=True)
        run_bounded_validation_calibration_queue(protocol, output, args.account_index, args.account_count,
                                                 args.maximum_cells, args.device, completed_directory)
    else:
        run_bounded_final_strategy_queue(protocol, output, args.account_index, args.account_count, args.maximum_cells, args.device)


if __name__ == "__main__":
    main()

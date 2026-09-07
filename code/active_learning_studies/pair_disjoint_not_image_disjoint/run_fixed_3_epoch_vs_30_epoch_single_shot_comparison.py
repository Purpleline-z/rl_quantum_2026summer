#!/usr/bin/env python3
"""Resumable, identity-safe fixed-3-versus-30-epoch single-shot comparison.

This runner deliberately holds learning rate fixed at 1e-4.  It compares the
historical three-epoch schedule with thirty epochs without using validation or
outer-test data to choose a training length.  Outer test is queried only at
the pre-registered reporting epochs (three and, where reached, thirty).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
RESULTS_ROOT = HERE / "results" / "simclr_three_seed_identity_safe_task3"
CELL_RESULTS_NAME = "fixed_epoch_3_and_30_single_shot_cells"
AGGREGATE_RESULTS_NAME = "fixed_epoch_3_and_30_single_shot_aggregate"
sys.path.insert(0, str(CODE_ROOT / "active_learning_program"))

from pairwise_active_learning_pipeline import Config, Experiment  # noqa: E402
from resumable_model_training import train_with_epoch_checkpoints  # noqa: E402

ZERO_AUDIT_FIELDS = (
    "exact_pair_overlap",
    "reference_test_identity_overlap",
    "utility_test_identity_overlap",
    "reference_utility_identity_overlap",
    "pairwise_image_identity_overlap_outer_test",
    "pairwise_image_identity_overlap_reference",
    "pairwise_image_identity_overlap_utility_validation",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT.parent, text=True).strip()
    except Exception:
        return "unavailable"


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def task_key(task: dict[str, Any]) -> tuple[int, int, str, int]:
    return int(task["seed"]), int(task["budget"]), str(task["strategy"]), int(task["epochs"])


def cell_filename(task: dict[str, Any]) -> str:
    return (f"seed_{task['seed']}_budget_{task['budget']}_strategy_{task['strategy']}"
            f"_fixed_{task['epochs']}_epochs.json")


def baseline_filename(seed: int, epochs: int) -> str:
    return f"seed_{seed}_initial_10_pair_groups_fixed_{epochs}_epochs.pth"


def all_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "fixed_epoch_single_shot_final",
            "seed": int(seed), "budget": int(budget), "strategy": str(strategy),
            "epochs": int(epochs), "learning_rate": float(protocol["learning_rate"]),
            "weight_decay": float(protocol["weight_decay"]),
            "outer_test_reporting_epochs": [value for value in protocol["outer_test_reporting_epochs"] if value <= epochs],
        }
        for seed in protocol["seeds"]
        for epochs in protocol["fixed_epoch_counts"]
        for budget in protocol["acquisition_budgets"]
        for strategy in protocol["strategies"]
    ]


def assigned_tasks(protocol: dict[str, Any], worker_index: int, worker_count: int) -> list[dict[str, Any]]:
    if worker_count != 2 or worker_index not in (0, 1):
        raise ValueError("This registered two-T4 study requires --worker-count 2 and --worker-index 0 or 1.")
    allowed = {tuple(values) for values in protocol["worker_assignment"][str(worker_index)]}
    return [task for task in all_tasks(protocol) if (task["seed"], task["epochs"]) in allowed]


def drive_cell_path(output: Path, task: dict[str, Any]) -> Path:
    return output / CELL_RESULTS_NAME / cell_filename(task)


def git_cell_exists(task: dict[str, Any]) -> bool:
    return any((RESULTS_ROOT / CELL_RESULTS_NAME).glob(f"account_*/{cell_filename(task)}"))


def make_experiment(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str, data_root: str | None) -> Experiment:
    cfg = Config(
        initial_pairs=int(protocol["initial_pair_groups"]), candidate_pairs=int(protocol["candidate_pair_groups"]),
        epochs=int(task["epochs"]), train_batch_size=int(protocol["train_batch_size"]),
        lr=float(task["learning_rate"]), weight_decay=float(task["weight_decay"]), seed=int(task["seed"]),
        device=device, dropout_p=float(protocol["dropout_probability"]), include_twinned=bool(protocol["include_twinned"]),
        encoder_initialization=str(protocol["encoder_initialization"]), acquisition_mode="single-shot",
        dataset_version=str(protocol["dataset_version"]), data_root=data_root,
        exclude_all_ideal_identities_from_pairwise=bool(protocol["exclude_all_ideal_identities_from_pairwise"]),
        manifest_dir=str(output / "identity_safe_split_audits" / f"seed_{task['seed']}"),
    )
    experiment = Experiment(cfg)
    experiment.output = output
    experiment.utility_cache_path = output / "unused_utility_cache.json"
    experiment.utility_cache = {}
    return experiment


def assert_strict_audit(audit: dict[str, Any], protocol: dict[str, Any], initial: list[str], candidates: list[str]) -> None:
    failures = {field: audit.get(field) for field in ZERO_AUDIT_FIELDS if audit.get(field, 0) != 0}
    if len(initial) != int(protocol["initial_pair_groups"]):
        failures["initial_pair_group_capacity"] = {"observed": len(initial), "required": protocol["initial_pair_groups"]}
    if len(candidates) != int(protocol["candidate_pair_groups"]):
        failures["candidate_pair_group_capacity"] = {"observed": len(candidates), "required": protocol["candidate_pair_groups"]}
    if audit.get("candidate_labels_hidden_from_selector") is not True:
        failures["candidate_labels_hidden_from_selector"] = audit.get("candidate_labels_hidden_from_selector")
    if failures:
        raise RuntimeError(f"Fixed-epoch identity-safe preflight failed: {failures}")


def train_recording_protocol(exp: Experiment, pair_ids: list[str], checkpoint: Path, phase: str,
                              outer_epochs: list[int], progress) -> tuple[Any, dict[str, Any], bool]:
    return train_with_epoch_checkpoints(
        exp, pair_ids, checkpoint, phase, None,
        heartbeat_seconds=1800, checkpoint_enabled=True, validation_split="utility_validation",
        early_stopping_patience=None, return_best_validation_model=False,
        evaluation_splits_by_epoch={epoch: ("outer_test",) for epoch in outer_epochs},
        progress_callback=progress,
    )


def metric_at(metrics: list[dict[str, Any]], epoch: int, split: str) -> dict[str, Any] | None:
    row = next((item for item in metrics if int(item["epoch"]) == epoch), None)
    if row is None:
        return None
    if split == "utility_validation":
        return {"accuracy": row.get("validation_accuracy")}
    return row.get("pre_registered_evaluations", {}).get(split)


def cleanup_finished_baselines(protocol: dict[str, Any], output: Path, tasks: list[dict[str, Any]]) -> None:
    for seed, epochs in {(int(item["seed"]), int(item["epochs"])) for item in tasks}:
        dependent = [item for item in tasks if int(item["seed"]) == seed and int(item["epochs"]) == epochs]
        if all(drive_cell_path(output, item).exists() or git_cell_exists(item) for item in dependent):
            (output / "resumable_checkpoints" / "shared_initial_selector_models" / baseline_filename(seed, epochs)).unlink(missing_ok=True)


def run_final_cell(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str,
                   data_root: str | None, progress) -> bool:
    destination = drive_cell_path(output, task)
    if destination.exists() or git_cell_exists(task):
        print(f"completed cell skipped: {cell_filename(task)}", flush=True)
        return False
    started = time.monotonic()
    exp = make_experiment(protocol, output / "runs" / cell_filename(task).removesuffix(".json"), task, device, data_root)
    initial, candidates = exp.load_and_split()
    audit = exp.protocol_audit(initial, candidates)
    assert_strict_audit(audit, protocol, initial, candidates)
    baseline_checkpoint = output / "resumable_checkpoints" / "shared_initial_selector_models" / baseline_filename(task["seed"], task["epochs"])
    baseline, baseline_metrics, paused = train_recording_protocol(
        exp, initial, baseline_checkpoint,
        f"initial-selector-seed-{task['seed']}-epochs-{task['epochs']}", [], progress,
    )
    if paused or baseline is None:
        raise RuntimeError("Initial selector training was interrupted; rerun the identical queue command.")
    candidate_rows, embedding_cache = exp.candidates_with_clusters(candidates, baseline)
    selected, _, _ = exp.select(task["strategy"], candidate_rows, baseline, embedding_cache, [],
                                budget=int(task["budget"]), labeled_ids=initial)
    selected_ids = [str(item["pair_id"]) for item in selected]
    if len(selected_ids) != int(task["budget"]) or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("Selector did not return the registered number of unique pair groups.")
    final_checkpoint = output / "resumable_checkpoints" / "final_cells" / cell_filename(task).replace(".json", ".pth")
    final_model, final_metrics, paused = train_recording_protocol(
        exp, initial + selected_ids, final_checkpoint,
        f"final-{cell_filename(task).removesuffix('.json')}", list(task["outer_test_reporting_epochs"]), progress,
    )
    if paused or final_model is None:
        raise RuntimeError("Final model training was interrupted; rerun the identical queue command.")
    epochs = list(final_metrics["epoch_metrics"])
    record = {
        **task,
        "git_sha": git_sha(), "initial_pair_group_ids": initial, "selected_pair_ids": selected_ids,
        "initial_pair_groups": len(initial), "candidate_pair_groups": len(candidates), "identity_audit": audit,
        "candidate_labels_read_during_selection": False,
        "baseline_epoch_metrics": baseline_metrics["epoch_metrics"],
        "final_epoch_metrics": epochs,
        "utility_validation_at_epoch_3": metric_at(epochs, 3, "utility_validation"),
        "utility_validation_at_epoch_30": metric_at(epochs, 30, "utility_validation"),
        "outer_test_at_epoch_3": metric_at(epochs, 3, "outer_test"),
        "outer_test_at_epoch_30": metric_at(epochs, 30, "outer_test"),
        "outer_test_not_used_for_selection": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(record, destination)
    final_checkpoint.unlink(missing_ok=True)
    clear_gpu_memory()
    print(f"saved completed cell: {destination}", flush=True)
    return True


def emit_state(output: Path, worker_index: int, total: int, completed: int, started: float, current: dict[str, Any]) -> None:
    elapsed = time.monotonic() - started
    width = 24
    filled = int(width * completed / total) if total else width
    print(f"[{'=' * filled}{'-' * (width - filled)}] worker={worker_index + 1}/2 completed={completed}/{total} "
          f"cell={current.get('cell', '-')} epoch={current.get('epoch', '-')}/{current.get('maximum_epochs', '-')} "
          f"elapsed={elapsed / 60:.1f}m", flush=True)
    atomic_json({"worker_index": worker_index, "completed_cells": completed, "total_cells": total,
                 "elapsed_seconds": elapsed, "current": current, "updated_unix": time.time()}, output / "run_state.json")


def run_automatic_queue(protocol: dict[str, Any], output: Path, device: str, worker_index: int,
                        worker_count: int, data_root: str | None) -> None:
    tasks = assigned_tasks(protocol, worker_index, worker_count)
    completed = sum(drive_cell_path(output, task).exists() or git_cell_exists(task) for task in tasks)
    print(f"automatic queue worker {worker_index + 1}/2: {completed}/{len(tasks)} cells already complete; "
          f"each completed cell saves JSON to {output / CELL_RESULTS_NAME}", flush=True)
    started, current = time.monotonic(), {}
    cleanup_finished_baselines(protocol, output, tasks)
    for number, task in enumerate(tasks, start=1):
        current.update(cell=cell_filename(task), epoch=0, maximum_epochs=task["epochs"])
        emit_state(output, worker_index, len(tasks), completed, started, current)
        changed = run_final_cell(
            protocol, output, task, device, data_root,
            lambda values: (current.update(values), emit_state(output, worker_index, len(tasks), completed, started, current)),
        )
        if changed:
            completed += 1
        cleanup_finished_baselines(protocol, output, tasks)
        emit_state(output, worker_index, len(tasks), completed, started, current)
    print(f"automatic queue complete: worker {worker_index + 1}/2; {completed}/{len(tasks)} cells durable on Drive.", flush=True)


def verify_identity_safe_input_protocol(protocol: dict[str, Any], output: Path, device: str, data_root: str | None) -> None:
    audits = []
    for seed in protocol["seeds"]:
        task = {"seed": seed, "budget": protocol["acquisition_budgets"][0], "strategy": "random",
                "epochs": protocol["fixed_epoch_counts"][0], "learning_rate": protocol["learning_rate"],
                "weight_decay": protocol["weight_decay"]}
        exp = make_experiment(protocol, output / "preflight_runs", task, device, data_root)
        initial, candidates = exp.load_and_split(); audit = exp.protocol_audit(initial, candidates)
        assert_strict_audit(audit, protocol, initial, candidates)
        audits.append({"seed": seed, "initial_pair_groups": len(initial), "candidate_pair_groups": len(candidates), "audit": audit})
        clear_gpu_memory()
    atomic_json({"status": "pass", "study": protocol["study_name"], "strict_all_ideal_identity_exclusion": True,
                 "audits": audits}, output / "preflight_identity_safe_capacity_audit.json")
    print(f"preflight passed for {len(audits)} seeds: {output / 'preflight_identity_safe_capacity_audit.json'}", flush=True)


def bootstrap_interval(values: np.ndarray, seed: int = 20260907, iterations: int = 10_000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(iterations, len(values)), replace=True), axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def load_git_cells() -> pd.DataFrame:
    paths = sorted((RESULTS_ROOT / CELL_RESULTS_NAME).glob("account_*/*.json"))
    if not paths:
        raise RuntimeError(f"No pushed result JSON files found under {RESULTS_ROOT / CELL_RESULTS_NAME}.")
    return pd.DataFrame([read_json(path) | {"source_path": str(path)} for path in paths])


def save_figure(fig, root: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def aggregate_completed_results(protocol: dict[str, Any]) -> None:
    cells = load_git_cells()
    expected = {task_key(task) for task in all_tasks(protocol)}
    observed = {task_key(row) for row in cells.to_dict("records")}
    duplicated = cells.duplicated(["seed", "budget", "strategy", "epochs"], keep=False)
    if duplicated.any():
        raise RuntimeError("Duplicate pushed fixed-epoch cell specifications found; resolve before aggregation.")
    if observed != expected:
        raise RuntimeError(f"Incomplete fixed-epoch study: found {len(observed)} unique cells, expected {len(expected)}.")
    destination = RESULTS_ROOT / AGGREGATE_RESULTS_NAME
    destination.mkdir(parents=True, exist_ok=True)
    compact = cells.drop(columns=[column for column in ("identity_audit", "baseline_epoch_metrics", "final_epoch_metrics") if column in cells], errors="ignore")
    compact.to_csv(destination / "per_seed_fixed_epoch_single_shot_results.csv", index=False)
    epoch_rows = []
    for row in cells.to_dict("records"):
        for metric in row["final_epoch_metrics"]:
            epoch_rows.append({"seed": row["seed"], "budget": row["budget"], "strategy": row["strategy"],
                               "fixed_epochs": row["epochs"], "epoch": metric["epoch"],
                               "train_loss": metric["train_loss"], "utility_validation_accuracy": metric.get("validation_accuracy")})
    epochs = pd.DataFrame(epoch_rows)
    epochs.to_csv(destination / "per_seed_epoch_training_and_validation_metrics.csv", index=False)
    summary = epochs.groupby(["fixed_epochs", "strategy", "budget", "epoch"], as_index=False).agg(
        mean_train_loss=("train_loss", "mean"), sd_train_loss=("train_loss", "std"),
        mean_utility_validation_accuracy=("utility_validation_accuracy", "mean"),
        sd_utility_validation_accuracy=("utility_validation_accuracy", "std"), seeds=("seed", "nunique"),
    )
    summary.to_csv(destination / "learning_dynamics_summary_by_epoch.csv", index=False)
    endpoint_rows = []
    for row in cells.to_dict("records"):
        for reporting_epoch in row["outer_test_reporting_epochs"]:
            result = row.get(f"outer_test_at_epoch_{reporting_epoch}")
            if result:
                endpoint_rows.append({"seed": row["seed"], "budget": row["budget"], "strategy": row["strategy"],
                                      "fixed_epochs": row["epochs"], "reporting_epoch": reporting_epoch,
                                      "outer_test_accuracy": result["accuracy"], "outer_test_correct": result["correct"],
                                      "outer_test_total": result["total"]})
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(destination / "pre_registered_outer_test_results_at_epochs_3_and_30.csv", index=False)
    endpoint_summary = endpoints.groupby(["fixed_epochs", "reporting_epoch", "strategy", "budget"], as_index=False).agg(
        mean_outer_test_accuracy=("outer_test_accuracy", "mean"), sd_outer_test_accuracy=("outer_test_accuracy", "std"), seeds=("seed", "nunique"),
    )
    endpoint_summary.to_csv(destination / "fixed_epoch_outer_test_summary.csv", index=False)
    random = endpoints[endpoints.strategy.eq("random")][["seed", "fixed_epochs", "reporting_epoch", "budget", "outer_test_accuracy"]].rename(columns={"outer_test_accuracy": "random_outer_test_accuracy"})
    paired = endpoints[endpoints.strategy.eq("uncertainty")].merge(random, on=["seed", "fixed_epochs", "reporting_epoch", "budget"], validate="one_to_one")
    paired["uncertainty_minus_random"] = paired.outer_test_accuracy - paired.random_outer_test_accuracy
    paired_rows = []
    for keys, group in paired.groupby(["fixed_epochs", "reporting_epoch", "budget"]):
        low, high = bootstrap_interval(group.uncertainty_minus_random.to_numpy())
        paired_rows.append({"fixed_epochs": keys[0], "reporting_epoch": keys[1], "budget": keys[2],
                            "paired_mean_difference": group.uncertainty_minus_random.mean(),
                            "paired_sd_difference": group.uncertainty_minus_random.std(ddof=1),
                            "bootstrap_95_ci_low": low, "bootstrap_95_ci_high": high, "seeds": len(group)})
    pd.DataFrame(paired_rows).to_csv(destination / "paired_uncertainty_minus_random_outer_test_difference.csv", index=False)
    figures = []
    for budget in protocol["acquisition_budgets"]:
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        subset = summary[summary.budget.eq(budget)]
        for (fixed_epochs, strategy), group in subset.groupby(["fixed_epochs", "strategy"]):
            label = f"{strategy}, fixed {fixed_epochs} epochs"
            axes[0].plot(group.epoch, group.mean_train_loss, marker="o", label=label)
            axes[1].plot(group.epoch, group.mean_utility_validation_accuracy, marker="o", label=label)
        axes[0].set(title=f"Budget {budget}: training dynamics", xlabel="Epoch", ylabel="Bradley–Terry train loss")
        axes[1].set(title=f"Budget {budget}: validation dynamics", xlabel="Epoch", ylabel="Utility-validation accuracy")
        for axis in axes:
            axis.grid(alpha=.25); axis.legend(fontsize=7)
        figure.tight_layout()
        figures += save_figure(figure, destination, f"budget_{budget}_fixed_3_vs_30_epoch_learning_dynamics")
    audits = pd.DataFrame([{"seed": row["seed"], **row["identity_audit"]} for row in cells.to_dict("records")]).groupby("seed", as_index=False).first()
    audits.to_csv(destination / "strict_identity_audit_by_seed.csv", index=False)
    pd.DataFrame({"file": [str(path.relative_to(destination)) for path in figures]}).to_csv(destination / "figure_manifest.csv", index=False)
    print(f"aggregated {len(cells)} cells into {destination}", flush=True)


def run_behavior_tests() -> None:
    module = "active_learning_program.code_behavior_tests.test_fixed_epoch_single_shot_comparison"
    result = subprocess.run([sys.executable, "-m", "unittest", module], cwd=CODE_ROOT)
    if result.returncode:
        raise RuntimeError("Fixed-epoch single-shot behavior tests failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify_identity_safe_input_protocol", "run_behavior_tests", "run_automatic_queue", "aggregate_completed_results"))
    parser.add_argument("--config", type=Path, default=HERE / "fixed_3_epoch_vs_30_epoch_single_shot_settings.json")
    parser.add_argument("--drive-output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--data-root")
    args = parser.parse_args()
    protocol = read_json(args.config)
    if args.action == "aggregate_completed_results":
        aggregate_completed_results(protocol)
        return
    if args.action == "run_behavior_tests":
        run_behavior_tests()
        return
    if args.drive_output is None:
        parser.error("--drive-output is required for preflight and GPU queue actions.")
    output = args.drive_output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.action == "verify_identity_safe_input_protocol":
        verify_identity_safe_input_protocol(protocol, output, args.device, args.data_root)
    else:
        if args.worker_index is None or args.worker_count is None:
            parser.error("--worker-index and --worker-count are required for run_automatic_queue.")
        run_automatic_queue(protocol, output, args.device, args.worker_index, args.worker_count, args.data_root)


if __name__ == "__main__":
    main()

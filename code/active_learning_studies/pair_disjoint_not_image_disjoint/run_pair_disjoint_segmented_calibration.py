#!/usr/bin/env python3
"""Drive-backed, pair-disjoint calibration tasks with bounded epoch segments.

This runner deliberately uses pair-group disjointness only: an unordered image
pair belongs to exactly one of the initial labelled set or candidate pool, but
an image may occur in more than one pair.  It contains no custom selector.
Every training segment checkpoints directly to ``--drive-output``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT / "active_learning_program"))

from pairwise_active_learning_pipeline import Config, Experiment
from resumable_model_training import train_with_epoch_checkpoints


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def task_id(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def experiment(protocol: dict[str, Any], output: Path, spec: dict[str, Any], target_epochs: int, device: str) -> Experiment:
    cfg = Config(
        initial_pairs=int(protocol["initial_pair_groups"]),
        candidate_pairs=int(protocol["maximum_acquired_pair_groups"]),
        epochs=target_epochs,
        train_batch_size=int(protocol["train_batch_size"]),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
        seed=int(spec["seed"]),
        device=device,
        dropout_p=float(protocol["dropout_probability"]),
        encoder_initialization=str(spec["encoder"]),
        acquisition_mode="single-shot",
        dataset_version=str(protocol["dataset_version"]),
        manifest_dir=str(output / "pair_disjoint_audits" / task_id(spec)),
    )
    value = Experiment(cfg)
    value.output = output
    value.utility_cache_path = output / "resumable_checkpoints" / "utility_cache_unused.json"
    value.utility_cache = {}
    return value


def audit_capacity(protocol: dict[str, Any], output: Path, device: str) -> None:
    rows: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        spec = {"seed": seed, "encoder": "simclr", "learning_rate": 1e-4, "weight_decay": protocol["default_weight_decay"]}
        value = experiment(protocol, output, spec, 3, device)
        initial, candidates = value.load_and_split()
        audit = value.protocol_audit(initial, candidates)
        rows.append({"seed": seed, "initial_pair_groups": len(initial), "candidate_pair_groups": len(candidates), "exact_pair_overlap": audit["exact_pair_overlap"], "image_overlap_initial_candidate": audit["image_overlap_initial_candidate"], "status": "pass" if len(initial) >= protocol["initial_pair_groups"] and len(candidates) >= protocol["maximum_acquired_pair_groups"] and audit["exact_pair_overlap"] == 0 else "fail"})
    path = output / "pair_disjoint_audits" / "capacity_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if any(row["status"] != "pass" for row in rows):
        raise RuntimeError(f"Pair-disjoint capacity audit failed. See {path}")


def write_calibration_manifest(protocol: dict[str, Any], output: Path) -> None:
    tasks = [{"seed": seed, "encoder": encoder, "learning_rate": lr, "weight_decay": protocol["default_weight_decay"], "total_epochs": epochs, "segment_epochs": protocol["maximum_epochs_per_gpu_segment"]} for seed in protocol["seeds"] for encoder in protocol["encoder_initializations"] for lr in protocol["learning_rates"] for epochs in protocol["epochs"]]
    write_json({"tasks": tasks, "rule": "Run one segment at a time; no segment exceeds 10 epochs."}, output / "task_manifests" / "pair_disjoint_calibration_tasks.json")
    print(f"Wrote {len(tasks)} calibration tasks. Each 30-epoch task has three separately resumable segments.", flush=True)


def calibration_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"seed": seed, "encoder": encoder, "learning_rate": lr, "weight_decay": protocol["default_weight_decay"], "total_epochs": epochs} for seed in protocol["seeds"] for encoder in protocol["encoder_initializations"] for lr in protocol["learning_rates"] for epochs in protocol["epochs"]]


def run_account_calibration_queue(protocol: dict[str, Any], output: Path, account_index: int, account_count: int, device: str) -> None:
    if not 0 <= account_index < account_count: raise ValueError("account_index must be from 0 through account_count - 1")
    assigned = [task for index, task in enumerate(calibration_tasks(protocol)) if index % account_count == account_index]
    print(f"account {account_index + 1}/{account_count}: {len(assigned)} calibration cells; every completed cell is saved to Drive", flush=True)
    for number, task in enumerate(assigned, start=1):
        identifier = task_id({key: task[key] for key in ("seed", "encoder", "learning_rate", "weight_decay")})
        result = output / "calibration_results" / f"{identifier}_epochs_{task['total_epochs']}.json"
        if result.exists():
            print(f"calibration queue {number}/{len(assigned)}: already complete {result.name}", flush=True); continue
        for segment in range(1, (int(task["total_epochs"]) + int(protocol["maximum_epochs_per_gpu_segment"]) - 1) // int(protocol["maximum_epochs_per_gpu_segment"]) + 1):
            run_calibration_segment(protocol, output, {key: task[key] for key in ("seed", "encoder", "learning_rate", "weight_decay")}, int(task["total_epochs"]), segment, device)
        print(f"calibration queue progress {number}/{len(assigned)} ({100 * number / len(assigned):.1f}%)", flush=True)


def lock_calibration_and_write_final_manifest(protocol: dict[str, Any], output: Path) -> None:
    rows = []
    for task in calibration_tasks(protocol):
        spec = {key: task[key] for key in ("seed", "encoder", "learning_rate", "weight_decay")}
        path = output / "calibration_results" / f"{task_id(spec)}_epochs_{task['total_epochs']}.json"
        if not path.exists(): raise RuntimeError(f"Calibration is incomplete: {path.name} is missing.")
        rows.append(read_json(path))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["encoder", "learning_rate", "weight_decay", "total_epochs"], as_index=False).agg(utility_validation_accuracy_mean=("utility_validation_accuracy", "mean"), utility_validation_accuracy_sd=("utility_validation_accuracy", "std"), outer_test_accuracy_mean=("outer_test_accuracy", "mean"), seeds=("seed", "nunique"))
    summary.to_csv(output / "calibration_results" / "calibration_summary.csv", index=False)
    locked: dict[str, dict[str, Any]] = {}
    for encoder, group in summary.groupby("encoder"):
        best = group.sort_values(["utility_validation_accuracy_mean", "total_epochs", "learning_rate"], ascending=[False, True, True]).iloc[0]
        locked[encoder] = {"learning_rate": float(best.learning_rate), "weight_decay": float(best.weight_decay), "epochs": int(best.total_epochs)}
    write_json({"locked_protocol_by_encoder": locked, "selection_endpoint": "utility_validation_accuracy_mean", "outer_test_not_used_for_selection": True}, output / "locked_calibration_protocol.json")
    strategies = ("random", "uncertainty", "core_set", "mc_dropout_mutual_information", "mc_dropout_probability_variance")
    tasks = [{"seed": seed, "encoder": encoder, "strategy": strategy, "budget": budget, **locked[encoder]} for seed in protocol["seeds"] for encoder in protocol["encoder_initializations"] for strategy in strategies for budget in protocol["budgets"]]
    write_json({"tasks": tasks, "expected_cells": len(tasks), "strategies": strategies, "rule": "Each cell trains a fresh baseline and fresh final model; completed JSON files are never overwritten."}, output / "task_manifests" / "pair_disjoint_final_budget_curve_tasks.json")
    print(f"Locked calibration protocol and wrote {len(tasks)} final strategy/budget cells.", flush=True)


def run_final_budget_curve_cell(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str) -> None:
    identifier = task_id(task); path = output / "final_budget_curve_cells" / f"{identifier}.json"
    if path.exists(): print(f"completed final cell: {path.name}", flush=True); return
    run_output = output / "final_budget_curve_runs" / identifier
    spec = {key: task[key] for key in ("seed", "encoder", "learning_rate", "weight_decay")}
    value = experiment(protocol, run_output, spec, int(task["epochs"]), device)
    initial, candidates = value.load_and_split()
    baseline, baseline_metrics = value.train(initial)
    candidate_rows, cache = value.candidates_with_clusters(candidates, baseline)
    selected, _, _ = value.select(task["strategy"], candidate_rows, baseline, cache, [], budget=int(task["budget"]), labeled_ids=initial)
    selected_ids = [row["pair_id"] for row in selected]
    final_model, final_metrics = value.train(initial + selected_ids)
    write_json({**task, "initial_pair_groups": len(initial), "selected_pair_groups": len(selected_ids), "selected_pair_ids": selected_ids, "baseline_training_pairwise_accuracy": baseline_metrics["pairwise_accuracy"], "final_training_pairwise_accuracy": final_metrics["pairwise_accuracy"], "utility_validation_accuracy": value.evaluate(final_model, "utility_validation")["test_accuracy"], "outer_test_accuracy": value.evaluate(final_model, "outer_test")["test_accuracy"]}, path)


def run_account_final_queue(protocol: dict[str, Any], output: Path, account_index: int, account_count: int, device: str) -> None:
    manifest = read_json(output / "task_manifests" / "pair_disjoint_final_budget_curve_tasks.json")
    assigned = [task for index, task in enumerate(manifest["tasks"]) if index % account_count == account_index]
    print(f"account {account_index + 1}/{account_count}: {len(assigned)} final cells; each result saves to Drive before the next begins", flush=True)
    for number, task in enumerate(assigned, start=1):
        run_final_budget_curve_cell(protocol, output, task, device)
        print(f"final queue progress {number}/{len(assigned)} ({100 * number / len(assigned):.1f}%)", flush=True)


def run_calibration_segment(protocol: dict[str, Any], output: Path, spec: dict[str, Any], total_epochs: int, segment_number: int, device: str) -> None:
    segment_epochs = int(protocol["maximum_epochs_per_gpu_segment"])
    target_epochs = min(total_epochs, segment_number * segment_epochs)
    if segment_number < 1 or target_epochs <= 0 or target_epochs - segment_epochs >= total_epochs:
        raise ValueError("segment_number is outside this task's epoch range")
    identifier = task_id(spec)
    completed = output / "calibration_results" / f"{identifier}_epochs_{total_epochs}.json"
    if completed.exists():
        print(f"completed calibration task: {completed}", flush=True)
        return
    value = experiment(protocol, output, spec, target_epochs, device)
    initial, _ = value.load_and_split()
    checkpoint = output / "resumable_checkpoints" / "pair_disjoint_calibration" / f"{identifier}.pth"
    model, metrics, paused = train_with_epoch_checkpoints(value, initial, checkpoint, f"pair-disjoint-calibration-{identifier}", None, heartbeat_seconds=int(protocol["checkpoint_heartbeat_seconds"]), checkpoint_enabled=True)
    if paused or model is None:
        raise RuntimeError("Training paused; rerun the identical segment command.")
    if target_epochs < total_epochs:
        write_json({"task": spec, "completed_epochs": target_epochs, "total_epochs": total_epochs, "checkpoint": str(checkpoint), "status": "segment_complete"}, output / "calibration_segment_status" / f"{identifier}.json")
        print(f"segment {segment_number} complete: {target_epochs}/{total_epochs} epochs saved to Drive. Run the next segment.", flush=True)
        return
    write_json({**spec, "total_epochs": total_epochs, "training_pairwise_accuracy": metrics["pairwise_accuracy"], "utility_validation_accuracy": value.evaluate(model, "utility_validation")["test_accuracy"], "outer_test_accuracy": value.evaluate(model, "outer_test")["test_accuracy"], "checkpoint": str(checkpoint)}, completed)
    print(f"final calibration result saved: {completed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit_pair_disjoint_capacity", "write_calibration_manifest", "run_account_calibration_queue", "lock_calibration_and_write_final_manifest", "run_account_final_queue", "run_calibration_segment"))
    parser.add_argument("--config", type=Path, default=HERE / "literature_backed_pair_disjoint_budget_curve_settings.json")
    parser.add_argument("--drive-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int); parser.add_argument("--encoder", choices=("simclr", "imagenet")); parser.add_argument("--learning-rate", type=float); parser.add_argument("--weight-decay", type=float); parser.add_argument("--total-epochs", type=int); parser.add_argument("--segment-number", type=int); parser.add_argument("--account-index", type=int); parser.add_argument("--account-count", type=int, default=4)
    args = parser.parse_args(); protocol = read_json(args.config); output = args.drive_output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.action == "audit_pair_disjoint_capacity": audit_capacity(protocol, output, args.device)
    elif args.action == "write_calibration_manifest": write_calibration_manifest(protocol, output)
    elif args.action == "run_account_calibration_queue":
        if args.account_index is None: parser.error("run_account_calibration_queue requires --account-index")
        run_account_calibration_queue(protocol, output, args.account_index, args.account_count, args.device)
    elif args.action == "lock_calibration_and_write_final_manifest": lock_calibration_and_write_final_manifest(protocol, output)
    elif args.action == "run_account_final_queue":
        if args.account_index is None: parser.error("run_account_final_queue requires --account-index")
        run_account_final_queue(protocol, output, args.account_index, args.account_count, args.device)
    else:
        needed = (args.seed, args.encoder, args.learning_rate, args.weight_decay, args.total_epochs, args.segment_number)
        if any(value is None for value in needed): parser.error("run_calibration_segment requires seed, encoder, learning-rate, weight-decay, total-epochs, and segment-number")
        spec = {"seed": args.seed, "encoder": args.encoder, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
        run_calibration_segment(protocol, output, spec, args.total_epochs, args.segment_number, args.device)


if __name__ == "__main__":
    main()

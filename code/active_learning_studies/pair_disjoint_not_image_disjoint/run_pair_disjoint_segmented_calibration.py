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
    parser.add_argument("action", choices=("audit_pair_disjoint_capacity", "write_calibration_manifest", "run_calibration_segment"))
    parser.add_argument("--config", type=Path, default=HERE / "literature_backed_pair_disjoint_budget_curve_settings.json")
    parser.add_argument("--drive-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int); parser.add_argument("--encoder", choices=("simclr", "imagenet")); parser.add_argument("--learning-rate", type=float); parser.add_argument("--weight-decay", type=float); parser.add_argument("--total-epochs", type=int); parser.add_argument("--segment-number", type=int)
    args = parser.parse_args(); protocol = read_json(args.config); output = args.drive_output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.action == "audit_pair_disjoint_capacity": audit_capacity(protocol, output, args.device)
    elif args.action == "write_calibration_manifest": write_calibration_manifest(protocol, output)
    else:
        needed = (args.seed, args.encoder, args.learning_rate, args.weight_decay, args.total_epochs, args.segment_number)
        if any(value is None for value in needed): parser.error("run_calibration_segment requires seed, encoder, learning-rate, weight-decay, total-epochs, and segment-number")
        spec = {"seed": args.seed, "encoder": args.encoder, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
        run_calibration_segment(protocol, output, spec, args.total_epochs, args.segment_number, args.device)


if __name__ == "__main__":
    main()

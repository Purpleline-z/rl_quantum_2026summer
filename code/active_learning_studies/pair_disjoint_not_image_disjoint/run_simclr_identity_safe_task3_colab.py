#!/usr/bin/env python3
"""Resumable identity-safe SimCLR Task 3 queue for a single Google Colab runtime.

Every command is idempotent: completed JSON cells are retained on Drive and an
interrupted epoch resumes from its checkpoint.  Calibration never opens the
outer test; only the frozen final queue reports it.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT / "active_learning_program"))
from pairwise_active_learning_pipeline import Config, Experiment  # noqa: E402
from resumable_model_training import train_with_epoch_checkpoints  # noqa: E402

STRATEGIES = (
    "random", "uncertainty", "core_set", "cluster_quota_uncertainty",
    "uncertainty_diversity", "cluster_margin_pairwise",
    "mc_dropout_probability_variance", "mc_dropout_mutual_information",
)
AUDIT_ZERO_FIELDS = (
    "exact_pair_overlap", "reference_test_identity_overlap", "utility_test_identity_overlap",
    "reference_utility_identity_overlap", "pairwise_image_identity_overlap_outer_test",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, path)


def stable_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT.parent, text=True).strip()
    except Exception:
        return "unavailable"


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MinuteProgress:
    """One-minute console/Drive heartbeat that remains useful in Colab logs."""
    def __init__(self, output: Path, phase: str, total: int, completed: int, estimate_seconds: float | None):
        self.output, self.phase, self.total, self.completed = output, phase, total, completed
        self.estimate_seconds, self.started, self.current = estimate_seconds, time.monotonic(), {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if self.estimate_seconds:
            print(f"Initial ETA: phase={self.phase}, total_cells={self.total}, estimated_duration={self.estimate_seconds * max(self.total-self.completed, 0)/3600:.2f}h, assumptions=warm-up cell timing", flush=True)
        else:
            print(f"ETA unavailable until first completed cell; phase={self.phase}, total_cells={self.total}", flush=True)
        self._thread.start()

    def update(self, **values: Any) -> None:
        self.current.update(values)

    def completed_cell(self, seconds: float) -> None:
        self.completed += 1
        remaining = self.total - self.completed
        self.estimate_seconds = seconds if self.estimate_seconds is None else .7 * self.estimate_seconds + .3 * seconds
        self.update(last_cell_seconds=seconds, eta_seconds=max(remaining, 0) * self.estimate_seconds)
        self.emit()

    def emit(self) -> None:
        elapsed = time.monotonic() - self.started
        eta = self.current.get("eta_seconds", max(self.total-self.completed, 0) * self.estimate_seconds if self.estimate_seconds else None)
        width, filled = 24, int(24 * self.completed / self.total) if self.total else 24
        gpu = f"{torch.cuda.memory_allocated()/2**30:.2f}GiB" if torch.cuda.is_available() else "CPU"
        message = (f"[{('=' * filled).ljust(width, '-')}] {self.completed}/{self.total} phase={self.phase} "
                   f"cell={self.current.get('cell', 'waiting')} epoch={self.current.get('epoch', '-')}/{self.current.get('maximum_epochs', '-')} "
                   f"elapsed={elapsed/60:.1f}m ETA={(eta/60 if eta is not None else float('nan')):.1f}m GPU={gpu}")
        print(message, flush=True)
        atomic_json({"phase": self.phase, "completed": self.completed, "total": self.total, "elapsed_seconds": elapsed,
                     "eta_seconds": eta, "current": self.current, "gpu_memory": gpu, "updated_unix": time.time()}, self.output / "run_state.json")

    def _loop(self) -> None:
        while not self._stop.wait(60):
            self.emit()

    def close(self) -> None:
        self.emit(); self._stop.set(); self._thread.join(timeout=2)


def wandb_run(project: str | None, config: dict[str, Any], name: str):
    if not project:
        return None
    try:
        import wandb
        return wandb.init(project=project, name=name, group=config.get("phase"), config=config, reinit=True)
    except Exception as error:
        print(f"W&B disabled for this cell: {error}", flush=True)
        return None


def make_experiment(protocol: dict[str, Any], output: Path, task: dict[str, Any], epochs: int, device: str) -> Experiment:
    selector = task.get("selector_parameters", {})
    cfg = Config(initial_pairs=int(protocol["initial_pair_groups"]), candidate_pairs=int(protocol["maximum_acquired_pair_groups"]),
                 epochs=epochs, train_batch_size=int(protocol["train_batch_size"]), lr=float(task["learning_rate"]),
                 weight_decay=float(protocol["weight_decay"]), seed=int(task["seed"]), device=device,
                 dropout_p=float(selector.get("dropout_probability", protocol["default_dropout_probability"])),
                 mc_samples=int(selector.get("mc_samples", protocol["default_mc_samples"])),
                 clusters=int(selector.get("cluster_count", protocol["default_cluster_count"])),
                 diversity_lambda=float(selector.get("diversity_lambda", protocol["default_diversity_lambda"])),
                 encoder_initialization="simclr", acquisition_mode="single-shot", dataset_version=protocol["dataset_version"],
                 manifest_dir=str(output / "identity_safe_split_audits" / stable_id(task)))
    exp = Experiment(cfg); exp.output = output; exp.utility_cache_path = output / "validation_utility_cache_unused.json"; exp.utility_cache = {}
    return exp


def assert_audit(exp: Experiment, initial: list[str], candidates: list[str]) -> dict[str, Any]:
    audit = exp.protocol_audit(initial, candidates)
    failures = {field: audit[field] for field in AUDIT_ZERO_FIELDS if audit.get(field, 0) != 0}
    if failures:
        raise RuntimeError(f"Identity-safe audit failed: {failures}")
    return audit


def calibration_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"phase": "task3a_validation", "seed": seed, "budget": budget, "learning_rate": lr, "epochs": epochs}
            for seed in protocol["seeds"] for budget in protocol["budgets"]
            for lr in protocol["learning_rates"] for epochs in protocol["maximum_epochs"]]


def result_path(output: Path, folder: str, task: dict[str, Any]) -> Path:
    return output / folder / f"{stable_id(task)}.json"


def deterministic_reference(candidates: list[str], seed: int, budget: int) -> list[str]:
    import random
    return random.Random(seed * 1_000_003 + budget).sample(candidates, min(budget, len(candidates)))


def baseline_id_for(task: dict[str, Any], protocol: dict[str, Any]) -> str:
    """Identify a reusable baseline without mixing dropout architectures."""
    key = {name: task[name] for name in ("seed", "budget", "learning_rate", "epochs")}
    key["dropout_probability"] = task.get("selector_parameters", {}).get("dropout_probability", protocol["default_dropout_probability"])
    return stable_id(key)


def cleanup_finished_shared_baselines(protocol: dict[str, Any], output: Path, tasks: list[dict[str, Any]], folder: str) -> None:
    """Remove only baselines whose every dependent durable result already exists."""
    for baseline_id in {baseline_id_for(task, protocol) for task in tasks}:
        dependent = [task for task in tasks if baseline_id_for(task, protocol) == baseline_id]
        if all(result_path(output, folder, task).exists() for task in dependent):
            (output / "resumable_checkpoints" / "shared_final_baselines" / f"{baseline_id}.pth").unlink(missing_ok=True)


def train_resumable(exp: Experiment, pairs: list[str], checkpoint: Path, phase: str, progress: MinuteProgress):
    return train_with_epoch_checkpoints(exp, pairs, checkpoint, phase, None, heartbeat_seconds=1800,
                                        validation_split="utility_validation", early_stopping_patience=5,
                                        progress_callback=lambda values: progress.update(**values))


def run_calibration_cell(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str, progress: MinuteProgress, project: str | None) -> None:
    target = result_path(output, "task3a_validation_cells", task)
    if target.exists(): return
    started = time.monotonic(); run = wandb_run(project, {**task, "git_sha": git_sha()}, stable_id(task))
    exp = make_experiment(protocol, output / "task3a_runs" / stable_id(task), task, int(task["epochs"]), device)
    initial, candidates = exp.load_and_split(); audit = assert_audit(exp, initial, candidates)
    selected = deterministic_reference(candidates, int(task["seed"]), int(task["budget"]))
    model, metrics, paused = train_resumable(exp, initial + selected, output / "resumable_checkpoints" / "task3a" / f"{stable_id(task)}.pth", stable_id(task), progress)
    if paused or model is None: raise RuntimeError("Calibration paused; repeat the same command.")
    validation = exp.evaluate(model, "utility_validation")
    record = {**task, "selector_parameters": None, "identity_audit": audit, "initial_pair_groups": len(initial),
              "reference_selected_pair_ids": selected, "utility_validation_accuracy": validation["test_accuracy"],
              "utility_validation_by_class": validation["by_class"], "best_validation_accuracy": metrics["best_validation_accuracy"],
              "epoch_metrics": metrics["epoch_metrics"], "outer_test_not_evaluated": True, "elapsed_seconds": time.monotonic()-started}
    atomic_json(record, target)
    if run:
        run.log({"utility_validation_accuracy": record["utility_validation_accuracy"], "elapsed_seconds": record["elapsed_seconds"]})
        run.save(str(target), policy="now"); run.finish()
    # The compact JSON is the durable record.  This checkpoint only exists for
    # interruption recovery and is not needed after a completed calibration cell.
    (output / "resumable_checkpoints" / "task3a" / f"{stable_id(task)}.pth").unlink(missing_ok=True)
    clear_memory()


def aggregate_calibration(protocol: dict[str, Any], output: Path) -> None:
    paths = list((output / "task3a_validation_cells").glob("*.json")); expected = calibration_tasks(protocol)
    if len(paths) != len(expected): raise RuntimeError(f"Calibration incomplete: found {len(paths)}, expected {len(expected)}.")
    rows = pd.DataFrame([read_json(path) for path in paths]); keys = ["seed", "budget", "learning_rate", "epochs"]
    if rows.duplicated(keys).any(): raise RuntimeError("Duplicate Task 3a cells found.")
    summary = rows.groupby(["budget", "learning_rate", "epochs"], as_index=False).agg(mean_validation_accuracy=("utility_validation_accuracy", "mean"), sd_validation_accuracy=("utility_validation_accuracy", "std"), n=("seed", "nunique"))
    locked: dict[str, dict[str, float | int]] = {}
    for budget, group in summary.groupby("budget"):
        best = group.sort_values(["mean_validation_accuracy", "epochs", "learning_rate"], ascending=[False, True, True]).iloc[0]
        locked[str(int(budget))] = {"learning_rate": float(best.learning_rate), "epochs": int(best.epochs)}
    destination = output / "frozen_task3b_protocol"; destination.mkdir(parents=True, exist_ok=True)
    rows.to_csv(destination / "three_seed_validation_calibration_cells.csv", index=False); summary.to_csv(destination / "three_seed_validation_calibration_summary.csv", index=False)
    atomic_json({"selection_endpoint": "mean utility-validation accuracy", "outer_test_used_for_selection": False,
                 "tie_break": "fewer epochs, then lower learning rate", "simclr_only": True, "protocol_by_budget": locked}, destination / "simclr_three_seed_budget_specific_schedule.json")


def selector_tasks(protocol: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    schedule = read_json(output / "frozen_task3b_protocol" / "simclr_three_seed_budget_specific_schedule.json")["protocol_by_budget"]["50"]
    tasks = []
    for seed in protocol["seeds"]:
        for value in protocol["diversity_lambdas"]:
            tasks.append({"phase": "selector_validation", "seed": seed, "budget": 50, "strategy": "uncertainty_diversity", **schedule, "selector_parameters": {"diversity_lambda": value}})
        for value in protocol["cluster_counts"]:
            for strategy in ("cluster_quota_uncertainty", "cluster_margin_pairwise"):
                tasks.append({"phase": "selector_validation", "seed": seed, "budget": 50, "strategy": strategy, **schedule, "selector_parameters": {"cluster_count": value}})
        for probability in protocol["dropout_probabilities"]:
            for samples in protocol["mc_samples"]:
                for strategy in ("mc_dropout_probability_variance", "mc_dropout_mutual_information"):
                    tasks.append({"phase": "selector_validation", "seed": seed, "budget": 50, "strategy": strategy, **schedule, "selector_parameters": {"dropout_probability": probability, "mc_samples": samples}})
    return tasks


def run_strategy_cell(protocol: dict[str, Any], output: Path, task: dict[str, Any], device: str, progress: MinuteProgress, project: str | None, folder: str, allow_outer_test: bool) -> None:
    target = result_path(output, folder, task)
    if target.exists(): return
    started = time.monotonic(); run = wandb_run(project, {**task, "git_sha": git_sha()}, stable_id(task))
    exp = make_experiment(protocol, output / f"{folder}_runs" / stable_id(task), task, int(task["epochs"]), device)
    initial, candidates = exp.load_and_split(); audit = assert_audit(exp, initial, candidates)
    baseline_id = baseline_id_for(task, protocol)
    baseline_checkpoint = output / "resumable_checkpoints" / "shared_final_baselines" / f"{baseline_id}.pth"
    baseline_model, _, paused = train_resumable(exp, initial, baseline_checkpoint, f"baseline-{baseline_id}", progress)
    if paused or baseline_model is None: raise RuntimeError("Baseline paused; repeat the same command.")
    candidate_rows, cache = exp.candidates_with_clusters(candidates, baseline_model)
    selected, _, _ = exp.select(task["strategy"], candidate_rows, baseline_model, cache, [], budget=int(task["budget"]), labeled_ids=initial)
    selected_ids = [item["pair_id"] for item in selected]
    final_checkpoint = output / "resumable_checkpoints" / folder / f"{stable_id(task)}.pth"
    model, metrics, paused = train_resumable(exp, initial + selected_ids, final_checkpoint, f"final-{stable_id(task)}", progress)
    if paused or model is None: raise RuntimeError("Final training paused; repeat the same command.")
    validation = exp.evaluate(model, "utility_validation")
    record = {**task, "identity_audit": audit, "selected_pair_ids": selected_ids,
              "utility_validation_accuracy": validation["test_accuracy"], "utility_validation_by_class": validation["by_class"],
              "best_validation_accuracy": metrics["best_validation_accuracy"], "epoch_metrics": metrics["epoch_metrics"],
              "elapsed_seconds": time.monotonic()-started, "outer_test_not_used_for_selection": True}
    if allow_outer_test:
        outer = exp.evaluate(model, "outer_test"); record.update(outer_test_accuracy=outer["test_accuracy"], outer_test_by_class=outer["by_class"])
    else: record["outer_test_not_evaluated"] = True
    atomic_json(record, target)
    if run:
        run.log({key: value for key, value in record.items() if isinstance(value, (int, float))})
        run.save(str(target), policy="now"); run.finish()
    # Preserve a final checkpoint only while its JSON result is not durable.
    # Shared baseline checkpoints are cleaned once every dependent strategy cell
    # is complete; they remain available for the next strategy in the meantime.
    final_checkpoint.unlink(missing_ok=True)
    clear_memory()


def aggregate_selector_screen(protocol: dict[str, Any], output: Path) -> None:
    paths = list((output / "selector_validation_cells").glob("*.json")); expected = selector_tasks(protocol, output)
    if len(paths) != len(expected): raise RuntimeError(f"Selector pre-screen incomplete: found {len(paths)}, expected {len(expected)}.")
    rows = pd.DataFrame([read_json(path) for path in paths])
    rows["parameters"] = rows.selector_parameters.map(lambda value: json.dumps(value, sort_keys=True))
    summary = rows.groupby(["strategy", "parameters"], as_index=False).agg(mean_validation_accuracy=("utility_validation_accuracy", "mean"), n=("seed", "nunique"))
    chosen: dict[str, Any] = {}
    for strategy, group in summary.groupby("strategy"):
        chosen[strategy] = json.loads(group.sort_values(["mean_validation_accuracy", "parameters"], ascending=[False, True]).iloc[0].parameters)
    # Use one agreed setting for both related strategies, retaining the requested global knobs.
    global_parameters = {"diversity_lambda": chosen["uncertainty_diversity"]["diversity_lambda"],
                         "cluster_count": chosen["cluster_quota_uncertainty"]["cluster_count"],
                         "dropout_probability": chosen["mc_dropout_probability_variance"]["dropout_probability"],
                         "mc_samples": chosen["mc_dropout_probability_variance"]["mc_samples"]}
    destination = output / "frozen_selector_parameters"; destination.mkdir(parents=True, exist_ok=True)
    rows.to_csv(destination / "three_seed_budget50_selector_screen_cells.csv", index=False); summary.to_csv(destination / "three_seed_budget50_selector_screen_summary.csv", index=False)
    atomic_json({"selection_endpoint": "mean utility-validation accuracy", "outer_test_used_for_selection": False,
                 "global_selector_parameters": global_parameters, "per_strategy_winners": chosen}, destination / "frozen_global_selector_parameters.json")


def final_tasks(protocol: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    schedule = read_json(output / "frozen_task3b_protocol" / "simclr_three_seed_budget_specific_schedule.json")["protocol_by_budget"]
    selector = read_json(output / "frozen_selector_parameters" / "frozen_global_selector_parameters.json")["global_selector_parameters"]
    return [{"phase": "task3c_final", "seed": seed, "budget": budget, "strategy": strategy, **schedule[str(budget)], "selector_parameters": selector}
            for seed in protocol["seeds"] for budget in protocol["budgets"] for strategy in STRATEGIES]


def run_queue(action: str, protocol: dict[str, Any], output: Path, device: str, project: str | None) -> None:
    tasks = calibration_tasks(protocol) if action == "run_task3a" else selector_tasks(protocol, output) if action == "run_selector_screen" else final_tasks(protocol, output)
    folder = "task3a_validation_cells" if action == "run_task3a" else "selector_validation_cells" if action == "run_selector_screen" else "task3c_final_strategy_cells"
    complete = sum(result_path(output, folder, task).exists() for task in tasks)
    if action != "run_task3a":
        cleanup_finished_shared_baselines(protocol, output, tasks, folder)
    timing = output / "warmup_timing_estimate.json"; estimate = read_json(timing).get("seconds_per_final_cell") if timing.exists() else None
    progress = MinuteProgress(output, action, len(tasks), complete, estimate); progress.start()
    try:
        for task in tasks:
            if result_path(output, folder, task).exists(): continue
            progress.update(cell=stable_id(task), epoch=0, maximum_epochs=task["epochs"])
            started = time.monotonic()
            if action == "run_task3a": run_calibration_cell(protocol, output, task, device, progress, project)
            else: run_strategy_cell(protocol, output, task, device, progress, project, folder, action == "run_task3c")
            if action != "run_task3a": cleanup_finished_shared_baselines(protocol, output, tasks, folder)
            progress.completed_cell(time.monotonic()-started)
    finally:
        progress.close()


def warmup(protocol: dict[str, Any], output: Path, device: str) -> None:
    task = {"phase": "warmup", "seed": protocol["seeds"][0], "budget": 10, "learning_rate": protocol["learning_rates"][2], "epochs": 1}
    started = time.monotonic(); exp = make_experiment(protocol, output / "warmup_run", task, 1, device)
    initial, candidates = exp.load_and_split(); audit = assert_audit(exp, initial, candidates)
    model, metrics, paused = train_with_epoch_checkpoints(exp, initial, output / "resumable_checkpoints" / "warmup.pth", "warmup", None, checkpoint_enabled=True, validation_split="utility_validation", early_stopping_patience=5)
    if paused or model is None: raise RuntimeError("Warm-up paused; repeat the command.")
    candidate_rows, cache = exp.candidates_with_clusters(candidates, model); exp.select("uncertainty", candidate_rows, model, cache, [], budget=10, labeled_ids=initial); exp.evaluate(model, "utility_validation")
    elapsed = time.monotonic()-started
    atomic_json({"seconds_per_epoch": elapsed, "seconds_per_final_cell": elapsed * 12, "identity_audit": audit, "metrics": metrics,
                 "assumptions": "one epoch warm-up multiplied by 12 to cover baseline, selection, final training, validation, and overhead"}, output / "warmup_timing_estimate.json")
    (output / "resumable_checkpoints" / "warmup.pth").unlink(missing_ok=True)
    total_cells = len(calibration_tasks(protocol)) + 45 + len(protocol["seeds"]) * len(protocol["budgets"]) * len(STRATEGIES)
    print(f"Initial ETA: phase=warmup_complete, total_cells={total_cells}, estimated_duration={elapsed*12*total_cells/3600:.2f}h, assumptions=representative one-epoch warm-up", flush=True)
    clear_memory()


def write_protocol(protocol: dict[str, Any], output: Path) -> None:
    lines = ["# SimCLR identity-safe three-seed Task 3 protocol", "", "## Frozen design", "", "- Seeds: 42, 79, 123", "- Encoder: SimCLR only", "- Budgets: 10, 25, 50, 75, 100", "- Calibration: utility-validation only; outer test is unavailable during Task 3a and selector screening.", "- Durable state: Google Drive results plus epoch checkpoints, 30-minute checkpoints, and minute ETA heartbeats.", "", "## Runtime provenance", "", f"- Git SHA at protocol generation: `{git_sha()}`", f"- Python: `{platform.python_version()}`", f"- Torch: `{torch.__version__}`", "", "## Commands", "", "See `COLAB_TASK3_SIMCLR_COMMANDS.md`; rerun any queue command unchanged after interruption."]
    (output / "EXPERIMENT_PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(protocol, output / "frozen_input_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write_protocol", "warmup", "run_task3a", "aggregate_task3b", "run_selector_screen", "aggregate_selector_screen", "run_task3c"))
    parser.add_argument("--config", type=Path, default=HERE / "simclr_three_seed_identity_safe_task3_settings.json")
    parser.add_argument("--drive-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--wandb-project")
    args = parser.parse_args(); protocol, output = read_json(args.config), args.drive_output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.action == "write_protocol": write_protocol(protocol, output)
    elif args.action == "warmup": warmup(protocol, output, args.device)
    elif args.action == "aggregate_task3b": aggregate_calibration(protocol, output)
    elif args.action == "aggregate_selector_screen": aggregate_selector_screen(protocol, output)
    else: run_queue(args.action, protocol, output, args.device, args.wandb_project)


if __name__ == "__main__":
    main()

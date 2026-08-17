#!/usr/bin/env python3
"""Create, run, resume, and aggregate bounded fixed-protocol validation jobs."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pairwise_active_learning_pipeline import Config, Experiment
from project_file_locations import RESULT_ROOT
from pair_acquisition_shared_calculations import (aggregate_rows, cluster_entropy, core_set_select, image_reuse_rate,
    mean_pair_cosine_similarity, pair_vector, uncertainty_diversity_select)
from resumable_model_training import atomic_torch_save, train_with_epoch_checkpoints
from pair_acquisition_methods import build_embedding_cache, cluster_quota_uncertainty_sampling, random_sampling, score_uncertainty, uncertainty_sampling

ROOT = Path(__file__).resolve().parent.parent
OUT = RESULT_ROOT / "fixed_protocol_3seed"
EPOCHS, SEEDS, LAMBDAS = [1, 2, 3, 5, 10], [42, 79, 123, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313], [.25, .5, .75]
SYMMETRY_MODE = "none"
DATA_ROOT: str | None = None
DATASET_VERSION = "v1.8"


def checkpointing_enabled(max_runtime_minutes: float) -> bool:
    """Large resumable state is reserved for jobs explicitly longer than two hours."""
    return max_runtime_minutes > 120


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str)); os.replace(temp, path)


def job_id(spec: dict[str, Any]) -> str:
    return "_".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in spec.items())


def manifest_path(stage: str) -> Path: return OUT / stage / "job_manifest.json"


def generate_manifest(stage: str, fixed_epoch: int | None = None, fixed_lambda: float | None = None, smoke: bool = False) -> Path:
    epochs = [1, 2] if smoke else EPOCHS; seeds = [42, 79] if smoke else SEEDS; budget = 4 if smoke else 100
    specs = []
    if stage == "epoch_sweep":
        specs = [{"stage": stage, "dataset_version": DATASET_VERSION, "epoch": epoch, "seed": seed, "strategy": strategy, "budget": budget} for epoch in epochs for seed in seeds for strategy in ("random", "uncertainty")]
    elif stage == "lambda_sweep":
        if fixed_epoch is None: fixed_epoch = chosen_value("epoch_sweep", "chosen_epoch")
        specs = [{"stage": stage, "dataset_version": DATASET_VERSION, "epoch": fixed_epoch, "seed": seed, "strategy": "uncertainty_diversity", "lambda": value, "budget": budget} for value in LAMBDAS for seed in seeds]
    elif stage == "final_comparison":
        if fixed_epoch is None: fixed_epoch = chosen_value("epoch_sweep", "chosen_epoch")
        if fixed_lambda is None: fixed_lambda = chosen_value("lambda_sweep", "chosen_lambda")
        strategies = ("random", "uncertainty", "cluster_quota_uncertainty", "uncertainty_diversity", "core_set")
        specs = [{"stage": stage, "dataset_version": DATASET_VERSION, "epoch": fixed_epoch, "seed": seed, "strategy": strategy, "lambda": fixed_lambda, "budget": budget} for seed in seeds for strategy in strategies]
    else: raise ValueError(stage)
    for i, spec in enumerate(specs): spec["job_id"] = f"{i:03d}_{job_id(spec)}"
    path = manifest_path(stage); atomic_json({"stage": stage, "jobs": specs}, path); print(path); return path


def chosen_value(stage: str, key: str):
    path = OUT / stage / "choice.json"
    if not path.exists(): raise FileNotFoundError(f"Run and aggregate {stage} before generating the next stage.")
    return json.loads(path.read_text())[key]


def load_spec(stage: str, requested: str) -> dict[str, Any]:
    specs = json.loads(manifest_path(stage).read_text())["jobs"]
    matches = [x for x in specs if x["job_id"] == requested or x["job_id"].startswith(requested)]
    if len(matches) != 1: raise ValueError(f"job-id {requested!r} did not identify exactly one job")
    return matches[0]


def config_for(spec: dict[str, Any]) -> Config:
    return Config(initial_pairs=50, candidate_pairs=120, budget=spec["budget"], batch_size=5, epochs=spec["epoch"], seed=spec["seed"], strategies=spec["strategy"], utility_per_pair=False, symmetry_mode=SYMMETRY_MODE, data_root=DATA_ROOT, dataset_version=spec["dataset_version"])


def pair_items(exp: Experiment, ids: list[str]) -> list[dict[str, Any]]:
    return [{"pair_id": pair_id, "img1": exp.groups[pair_id].iloc[0].resolved_img1, "img2": exp.groups[pair_id].iloc[0].resolved_img2} for pair_id in ids]


def candidate_data(exp: Experiment, pool: list[str], labeled: list[str], model):
    candidates, labeled_items = pair_items(exp, pool), pair_items(exp, labeled)
    cache = build_embedding_cache(candidates + labeled_items, model, exp.device)
    vectors = np.stack([pair_vector(item, cache) for item in candidates]); k = min(exp.cfg.clusters, len(candidates))
    labels = KMeans(n_clusters=k, random_state=exp.cfg.seed, n_init="auto").fit_predict(vectors) if k > 1 else np.zeros(len(candidates), dtype=int)
    for item, vector, cluster in zip(candidates, vectors, labels): item.update(pair_vector=vector.tolist(), cluster1=int(cluster))
    return candidates, np.stack([pair_vector(item, cache) for item in labeled_items]), cache


def select(spec: dict[str, Any], exp: Experiment, candidates, labeled_vectors, model, cache):
    budget, strategy = min(spec["budget"], len(candidates)), spec["strategy"]
    if strategy == "random": return random_sampling(candidates, model, budget, exp.device, exp.cfg.seed, cache)
    if strategy == "uncertainty": return uncertainty_sampling(candidates, model, budget, exp.device, exp.cfg.seed, cache)
    if strategy == "cluster_quota_uncertainty": return cluster_quota_uncertainty_sampling(candidates, model, budget, exp.device, exp.cfg.seed, cache)
    if strategy == "uncertainty_diversity":
        scored = score_uncertainty(candidates, model, exp.device, cache)
        return uncertainty_diversity_select(scored, labeled_vectors, budget, float(spec["lambda"]))
    if strategy == "core_set": return core_set_select(candidates, labeled_vectors, budget)
    raise ValueError(strategy)


def state_path(spec: dict[str, Any]) -> Path: return OUT / spec["stage"] / "jobs" / spec["job_id"] / "job_state.json"


def run_job(spec: dict[str, Any], resume: bool, max_minutes: float, checkpoint_minutes: float) -> None:
    directory = state_path(spec).parent; directory.mkdir(parents=True, exist_ok=True); state_file = state_path(spec)
    checkpoints = checkpointing_enabled(max_minutes)
    if resume and not checkpoints:
        raise ValueError("--resume requires --max-runtime-minutes greater than 120; shorter jobs intentionally do not save checkpoints.")
    state = json.loads(state_file.read_text()) if state_file.exists() else {"spec": spec, "status": "new", "completed_phases": []}
    if state.get("status") == "completed": print(f"{spec['job_id']}: already completed"); return
    if not checkpoints and state.get("status") != "new":
        state = {"spec": spec, "status": "new", "completed_phases": [], "restart_reason": "checkpointing disabled for jobs of two hours or less"}
    state["checkpointing_enabled"] = checkpoints
    deadline, heartbeat = time.monotonic() + max_minutes * 60, max(1, int(checkpoint_minutes * 60))
    exp = Experiment(config_for(spec)); exp.output = directory; exp.utility_cache_path = directory / "utility_cache.json"; exp.utility_cache = {}
    initial, pool = exp.load_and_split(); selected = state.get("selected", [])
    state["protocol_audit"] = exp.protocol_audit(initial, pool); atomic_json(state, state_file)
    def pause(phase: str):
        message = None if checkpoints else "Checkpointing is disabled for jobs of two hours or less; rerun starts training from the beginning."
        state.update(status="paused", phase=phase, pause_message=message); atomic_json(state, state_file); print(f"{spec['job_id']}: paused at {phase}" + (f" ({message})" if message else ""), flush=True)
    if "initial_train" not in state["completed_phases"]:
        state.update(status="running", phase="initial_train"); atomic_json(state, state_file)
        model, metrics, paused = train_with_epoch_checkpoints(exp, initial, directory / "initial_epoch_checkpoint.pth", "initial_train", deadline, heartbeat, checkpoints)
        if paused: return pause("initial_train")
        state.update(initial_metrics=metrics, initial_eval=exp.evaluate(model)); state["completed_phases"].append("initial_train"); atomic_json(state, state_file)
    else:
        checkpoint = torch_load(directory / "initial_epoch_checkpoint.pth", exp.device); model = exp.make_model(); model.load_state_dict(checkpoint["model"])
    if "selection" not in state["completed_phases"]:
        if time.monotonic() >= deadline: return pause("selection")
        print(f"heartbeat job={spec['job_id']} phase=selection", flush=True)
        candidates, labeled_vectors, cache = candidate_data(exp, pool, initial, model); selected_items = select(spec, exp, candidates, labeled_vectors, model, cache)
        selected = [x["pair_id"] for x in selected_items]
        state.update(selected=selected, selected_details=selected_items, selection_metrics={"cluster_coverage": len({x["cluster1"] for x in selected_items}), "cluster_entropy": cluster_entropy(selected_items), "image_reuse_rate": image_reuse_rate(selected_items), "mean_pair_cosine_similarity": mean_pair_cosine_similarity(selected_items)}); state["completed_phases"].append("selection"); atomic_json(state, state_file)
    if "post_train" not in state["completed_phases"]:
        model, metrics, paused = train_with_epoch_checkpoints(exp, initial + selected, directory / "post_epoch_checkpoint.pth", "post_train", deadline, heartbeat, checkpoints)
        if paused: return pause("post_train")
        state.update(post_metrics=metrics, post_eval=exp.evaluate(model)); state["completed_phases"].append("post_train"); atomic_json(state, state_file)
    if "report" not in state["completed_phases"]:
        pre, post = state["initial_eval"], state["post_eval"]
        result = {**spec, "protocol_audit": state["protocol_audit"], "checkpointing_enabled": checkpoints, "pre_test_accuracy": pre["test_accuracy"], "post_test_accuracy": post["test_accuracy"], "batch_utility": post["test_accuracy"] - pre["test_accuracy"], "pre_pairwise_accuracy": state["initial_metrics"]["pairwise_accuracy"], "post_pairwise_accuracy": state["post_metrics"]["pairwise_accuracy"], "selected_pairs": len(selected), "selected_rows": int(len(exp.rows_for(selected))), **state["selection_metrics"], "per_class_delta": {key: post["by_class"][key]["accuracy"] - pre["by_class"][key]["accuracy"] for key in pre["by_class"]}, "individual_utilities": state.get("individual_utilities")}
        atomic_json(result, directory / "result.json"); state.update(status="completed", result_path=str(directory / "result.json")); state["completed_phases"].append("report"); atomic_json(state, state_file)
        print(f"{spec['job_id']}: completed batch utility={result['batch_utility']:.4f}", flush=True)


def torch_load(path: Path, device):
    import torch
    return torch.load(path, map_location=device)


def completed_results(stage: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in (OUT / stage / "jobs").glob("*/result.json")]


def bootstrap_mean_ci(values: pd.Series, seed: int = 42, draws: int = 10_000) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    if not len(values): return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def aggregate(stage: str) -> None:
    rows = completed_results(stage)
    if not rows: raise ValueError("No completed jobs to aggregate.")
    keys = ["epoch", "strategy"] + (["lambda"] if stage == "lambda_sweep" else [])
    metrics = ["post_test_accuracy", "batch_utility", "post_pairwise_accuracy", "cluster_coverage", "image_reuse_rate", "mean_pair_cosine_similarity"]
    summary = aggregate_rows(rows, keys, metrics); directory = OUT / stage; pd.DataFrame(rows).to_csv(directory / "per_seed_results.csv", index=False); pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False)
    raw = pd.DataFrame(rows)
    frame = pd.DataFrame(summary)
    ci_rows = []
    for grouped_keys, group in raw.groupby(keys):
        grouped_keys = (grouped_keys,) if not isinstance(grouped_keys, tuple) else grouped_keys
        for metric in ("post_test_accuracy", "batch_utility"):
            low, high = bootstrap_mean_ci(group[metric], seed=42)
            ci_rows.append({**dict(zip(keys, grouped_keys)), "metric": metric, "bootstrap_ci_low": low, "bootstrap_ci_high": high})
    pd.DataFrame(ci_rows).to_csv(directory / "bootstrap_mean_ci.csv", index=False)
    if stage == "final_comparison" and "random" in set(raw.strategy):
        baseline = raw[raw.strategy == "random"].set_index("seed")
        paired = []
        for _, row in raw[raw.strategy != "random"].iterrows():
            if row.seed in baseline.index:
                ref = baseline.loc[row.seed]
                paired.append({"seed": int(row.seed), "strategy": row.strategy,
                               "post_accuracy_difference_vs_random": row.post_test_accuracy - ref.post_test_accuracy,
                               "batch_utility_difference_vs_random": row.batch_utility - ref.batch_utility})
        pd.DataFrame(paired).to_csv(directory / "paired_strategy_differences_vs_random.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    xkey = "lambda" if stage == "lambda_sweep" else "epoch"
    for strategy, group in frame.groupby("strategy"):
        group = group.sort_values(xkey); ax.errorbar(group[xkey], group["post_test_accuracy_mean"], yerr=group["post_test_accuracy_std"], marker="o", capsize=3, label=strategy)
    ax.set(xlabel=xkey, ylabel="Post-acquisition fixed-test accuracy", title=f"{stage} mean ± sample std"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(directory / "selection_plot.png", dpi=160); plt.close(fig)
    if stage == "epoch_sweep":
        uncertainty = frame[frame.strategy == "uncertainty"].sort_values("epoch"); best = uncertainty.loc[uncertainty.post_test_accuracy_mean.idxmax()]; overlapping = uncertainty[uncertainty.post_test_accuracy_mean + uncertainty.post_test_accuracy_std >= best.post_test_accuracy_mean - best.post_test_accuracy_std]
        choice = int(overlapping.epoch.min())
        atomic_json({"chosen_epoch": choice, "rule": "highest uncertainty mean; smallest epoch whose ±1 std interval overlaps best"}, directory / "choice.json")
    if stage == "lambda_sweep":
        ordered = frame.sort_values(["post_test_accuracy_mean", "batch_utility_mean", "lambda"], ascending=[False, False, True]); atomic_json({"chosen_lambda": float(ordered.iloc[0]["lambda"]), "rule": "post accuracy, then batch utility, then smaller lambda"}, directory / "choice.json")


def main():
    global OUT, SEEDS, SYMMETRY_MODE, DATA_ROOT, DATASET_VERSION
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--stage", choices=["epoch_sweep", "lambda_sweep", "final_comparison"], required=True); p.add_argument("--generate-manifest", action="store_true"); p.add_argument("--job-id"); p.add_argument("--aggregate", action="store_true"); p.add_argument("--resume", action="store_true", help="Requires --max-runtime-minutes greater than 120."); p.add_argument("--max-runtime-minutes", type=float, default=50, help="Checkpointing is enabled only above 120 minutes; default 50 writes no large .pth files."); p.add_argument("--checkpoint-minutes", type=float, default=1); p.add_argument("--smoke", action="store_true"); p.add_argument("--output-root", type=Path); p.add_argument("--data-root"); p.add_argument("--dataset-version", choices=["v1.8", "v5.7"], default="v1.8"); p.add_argument("--seeds"); p.add_argument("--symmetry-mode", choices=["none", "left_half_mirror", "symmetric_average"], default="none")
    a = p.parse_args()
    DATASET_VERSION = a.dataset_version
    OUT = a.output_root.resolve() if a.output_root else RESULT_ROOT / "fixed_protocol_audit" / DATASET_VERSION
    DATA_ROOT = a.data_root
    SYMMETRY_MODE = a.symmetry_mode
    if a.seeds: SEEDS = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    if a.generate_manifest: generate_manifest(a.stage, smoke=a.smoke)
    if a.job_id: run_job(load_spec(a.stage, a.job_id), a.resume, a.max_runtime_minutes, a.checkpoint_minutes)
    if a.aggregate: aggregate(a.stage)


if __name__ == "__main__": main()

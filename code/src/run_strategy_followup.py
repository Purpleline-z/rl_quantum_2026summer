#!/usr/bin/env python3
"""Fifteen-seed endpoint extension and evidence-first strategy analyses."""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import run_fixed_protocol_validation as validation
from selection_utils import aggregate_rows
from project_paths import RESULT_ROOT

ROOT = Path(__file__).resolve().parent.parent
RESULT = RESULT_ROOT; OUT = RESULT / "strategy_followup_analysis"
SEEDS = [42, 79, 123, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313]
STRATEGIES = ["random", "uncertainty", "cluster_quota_uncertainty", "uncertainty_diversity", "core_set"]


def atomic_json(value: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str)); os.replace(tmp, path)


def legacy_roots() -> list[Path]:
    return [RESULT / "fixed_protocol_3seed"]


def results_from(roots: list[Path]) -> list[tuple[dict, Path]]:
    found = []
    for root in roots:
        for file in root.glob("final_comparison/jobs/*/result.json"):
            try: found.append((json.loads(file.read_text()), file))
            except json.JSONDecodeError: pass
    return found


def baseline_config(previous: list[tuple[dict, Path]]) -> tuple[int, float, int]:
    if not previous: return 3, .5, 100
    r = previous[0][0]; return int(r.get("epoch", 3)), float(r.get("lambda", .5)), int(r.get("budget", 100))


def generate_manifest():
    old = results_from(legacy_roots()); current = results_from([OUT]); epoch, lam, budget = baseline_config(old or current)
    completed = {(int(r.get("seed")), r.get("strategy")) for r, _ in old + current}
    jobs = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            if (seed, strategy) not in completed:
                spec = {"stage": "final_comparison", "epoch": epoch, "seed": seed, "strategy": strategy, "lambda": lam, "budget": budget}
                spec["job_id"] = f"followup_seed-{seed}_strategy-{strategy}"; jobs.append(spec)
    source = [str(p) for p in legacy_roots()]
    atomic_json({"source_result_roots": source, "fixed_epoch": epoch, "fixed_lambda": lam, "budget": budget, "required_seeds": SEEDS, "jobs": jobs}, OUT / "manifests" / "missing_jobs_15_seed.json")
    print(f"Wrote {len(jobs)} missing jobs to {OUT / 'manifests' / 'missing_jobs_15_seed.json'}")


def run_job(job_id: str, resume: bool, max_minutes: float, checkpoint_minutes: float):
    manifest = json.loads((OUT / "manifests" / "missing_jobs_15_seed.json").read_text())
    hits = [x for x in manifest["jobs"] if x["job_id"] == job_id]
    if len(hits) != 1: raise ValueError(f"Unknown job {job_id}")
    validation.OUT = OUT
    validation.run_job(hits[0], resume, max_minutes, checkpoint_minutes)


def all_results() -> list[tuple[dict, Path]]:
    return results_from(legacy_roots() + [OUT])


def aggregate():
    rows = []
    seen = set()
    for r, path in all_results():
        key = (r.get("seed"), r.get("strategy"))
        if key in seen: continue
        seen.add(key); rows.append({**r, "source_result": str(path)})
    if not rows: raise ValueError("No completed final-comparison results were found.")
    out = OUT / "fifteen_seed_extension"; out.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(rows).to_csv(out / "fifteen_seed_per_run.csv", index=False)
    summary = aggregate_rows(rows, ["strategy"], ["pre_test_accuracy", "post_test_accuracy", "batch_utility", "post_pairwise_accuracy", "cluster_coverage", "image_reuse_rate", "mean_pair_cosine_similarity"])
    pd.DataFrame(summary).to_csv(out / "fifteen_seed_summary.csv", index=False)
    frame = pd.DataFrame(rows); paired = []
    for strategy in STRATEGIES:
        if strategy == "random": continue
        both = frame[frame.strategy == "random"].merge(frame[frame.strategy == strategy], on="seed", suffixes=("_random", "_strategy"))
        for _, row in both.iterrows(): paired.append({"seed": row.seed, "strategy": strategy, "post_accuracy_difference_vs_random": row.post_test_accuracy_strategy - row.post_test_accuracy_random, "batch_utility_difference_vs_random": row.batch_utility_strategy - row.batch_utility_random})
    pairs = pd.DataFrame(paired); pairs.to_csv(out / "paired_strategy_differences.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4)); sf = pd.DataFrame(summary)
    ax.bar(sf.strategy, sf.post_test_accuracy_mean, yerr=sf.post_test_accuracy_std, capsize=3); ax.set(ylabel="Post-acquisition accuracy", title="Fifteen-seed mean ± sample std (interim until complete)"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(out / "fifteen_seed_strategy_comparison.png", dpi=170); fig.savefig(out / "fifteen_seed_strategy_comparison.pdf"); plt.close(fig)
    if not pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 4));
        for strategy, group in pairs.groupby("strategy"): ax.scatter(group.seed, group.post_accuracy_difference_vs_random, label=strategy)
        ax.axhline(0, color="black", lw=1); ax.set(xlabel="Seed", ylabel="Post-accuracy difference vs random", title="Paired seed differences"); ax.legend(); fig.tight_layout(); fig.savefig(out / "paired_differences.png", dpi=170); fig.savefig(out / "paired_differences.pdf"); plt.close(fig)
    print(out)


def states() -> list[tuple[dict, Path]]:
    output = []
    for root in legacy_roots() + [OUT]:
        for file in root.glob("**/jobs/*/job_state.json"):
            try: output.append((json.loads(file.read_text()), file))
            except json.JSONDecodeError: pass
    return output


def analyze_coreset():
    output = OUT / "core_set_analysis"; output.mkdir(parents=True, exist_ok=True); rows = []
    for state, path in states():
        spec = state.get("spec", {}); details = state.get("selected_details", [])
        for item in details:
            vector = item.get("pair_vector")
            rows.append({"seed": spec.get("seed"), "strategy": spec.get("strategy"), "pair_id": item.get("pair_id"), "cluster": item.get("cluster1"), "core_set_distance": item.get("core_set_min_distance"), "uncertainty": item.get("uncertainty"), "vector_norm": float(np.linalg.norm(vector)) if vector else np.nan, "source_state": str(path)})
    table = pd.DataFrame(rows); table.to_csv(output / "core_set_selection_diagnostics.csv", index=False)
    if table.empty: (output / "pca_status.txt").write_text("No stored selection embeddings found; PCA comparison unavailable and no conclusion is claimed."); return
    cluster = table.groupby(["strategy", "cluster"], dropna=False).agg(selected_pairs=("pair_id", "count"), mean_core_set_distance=("core_set_distance", "mean"), mean_uncertainty=("uncertainty", "mean"), mean_norm=("vector_norm", "mean")).reset_index()
    cluster["low_value_flag"] = False  # utility/reconstruction evidence is unavailable unless recorded; do not fabricate flags.
    cluster.to_csv(output / "core_set_cluster_statistics.csv", index=False)
    vectors, labels = [], []
    for state, _ in states():
        for item in state.get("selected_details", []):
            if item.get("pair_vector") is not None: vectors.append(item["pair_vector"]); labels.append((state.get("spec", {}).get("strategy"), item.get("cluster1")))
    if len(vectors) >= 2:
        xy = PCA(n_components=2, random_state=0).fit_transform(np.asarray(vectors)); fig, ax = plt.subplots(figsize=(6, 5))
        for strategy in sorted({x[0] for x in labels}):
            idx = [i for i, x in enumerate(labels) if x[0] == strategy]; ax.scatter(xy[idx, 0], xy[idx, 1], s=12, label=strategy)
        ax.set(title="Stored selected pair embeddings: PCA", xlabel="PC1", ylabel="PC2"); ax.legend(); fig.tight_layout(); fig.savefig(output / "pca_selected_pairs.png", dpi=170); fig.savefig(output / "pca_selected_pairs.pdf"); plt.close(fig)
    (output / "interpretation.md").write_text("PCA is a global two-dimensional projection; core-set optimizes local Euclidean coverage in the full pair-embedding space. These are not contradictory without quantitative evidence that core-set distance correlates with reconstruction relevance or controlled utility.")


def analyze_redundancy():
    output = OUT / "redundancy_analysis"; output.mkdir(parents=True, exist_ok=True); rows = []
    for result, path in all_results():
        values = result.get("individual_utilities") or []
        utility = [x.get("delta_accuracy", x) if isinstance(x, dict) else x for x in values]
        rows.append({"seed": result.get("seed"), "strategy": result.get("strategy"), "pre_test_accuracy": result.get("pre_test_accuracy"), "post_test_accuracy": result.get("post_test_accuracy"), "batch_utility": result.get("batch_utility"), "mean_individual_utility": float(np.mean(utility)) if utility else np.nan, "sum_individual_utility": float(np.sum(utility)) if utility else np.nan, "interaction_gap": result.get("batch_utility") - float(np.sum(utility)) if utility else np.nan, "image_reuse_rate": result.get("image_reuse_rate"), "cluster_coverage": result.get("cluster_coverage"), "mean_pair_cosine_similarity": result.get("mean_pair_cosine_similarity"), "source_result": str(path)})
    frame = pd.DataFrame(rows); frame.to_csv(output / "redundancy_utility_accuracy.csv", index=False)
    for metric in ("image_reuse_rate", "mean_pair_cosine_similarity"):
        fig, ax = plt.subplots(figsize=(5, 4))
        for strategy, group in frame.groupby("strategy"): ax.scatter(group[metric], group.batch_utility, label=strategy)
        ax.set(xlabel=metric, ylabel="Within-seed batch utility", title="Redundancy versus utility"); ax.legend(); fig.tight_layout(); fig.savefig(output / f"{metric}_vs_batch_utility.png", dpi=170); fig.savefig(output / f"{metric}_vs_batch_utility.pdf"); plt.close(fig)
    report = "# Follow-up report\n\nThis analysis reports post-acquisition accuracy and within-seed batch utility separately. If a diversity method lowers reuse/similarity without improving either quantity, redundancy alone is not a sufficient explanation. Results are interim until all fifteen fixed seeds complete; report effect directions and paired differences, not definitive significance claims.\n"
    (OUT / "reports").mkdir(parents=True, exist_ok=True); (OUT / "reports" / "followup_report.md").write_text(report)


def main():
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["generate-manifest", "run-job", "aggregate", "analyze-coreset", "analyze-redundancy", "all-analysis"]); p.add_argument("--job-id"); p.add_argument("--resume", action="store_true"); p.add_argument("--max-runtime-minutes", type=float, default=50); p.add_argument("--checkpoint-minutes", type=float, default=1); p.add_argument("--data-root"); a = p.parse_args()
    validation.DATA_ROOT = a.data_root
    if a.action == "generate-manifest": generate_manifest()
    elif a.action == "run-job": run_job(a.job_id, a.resume, a.max_runtime_minutes, a.checkpoint_minutes)
    elif a.action == "aggregate": aggregate()
    elif a.action == "analyze-coreset": analyze_coreset()
    elif a.action == "analyze-redundancy": analyze_redundancy()
    else: aggregate(); analyze_coreset(); analyze_redundancy()


if __name__ == "__main__": main()

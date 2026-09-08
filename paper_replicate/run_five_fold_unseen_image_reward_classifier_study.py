"""One-T4, five-fold evaluation of a frozen-SimCLR static RHEED reward classifier."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from .evaluate_image_disjoint_reconstruction_reward_model import evaluate_sealed_test_set, fit_validation_calibration
from .export_active_learning_pair_selection_features import export_pair_selection_features
from .pairwise_and_absolute_label_dataset import create_fixed_five_fold_unseen_image_splits, load_pairwise_rows
from .train_peak_aware_reconstruction_reward_model import train_job


VARIANTS = ("image_encoder_only", "image_encoder_plus_peak_features")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def bootstrap_interval(values, seed=42, repetitions=4000):
    values = [value for value in values if value is not None]
    if not values: return {"mean": None, "sample_standard_deviation": None, "bootstrap_95_percent_interval": None}
    generator = random.Random(seed)
    samples = sorted(sum(generator.choice(values) for _ in values) / len(values) for _ in range(repetitions))
    return {"mean": mean(values), "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.,
            "bootstrap_95_percent_interval": [samples[int(.025 * (repetitions - 1))], samples[int(.975 * (repetitions - 1))]],
            "unit": "outer-fold mean across three training seeds", "sample_count": len(values)}


def _source_commit(repository_root):
    if not repository_root: return None
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root, capture_output=True, text=True, check=True).stdout.strip()


def _load_or_train(data_root, split, output, variant, seed, device, epochs, resume):
    complete = output / "completed_task_result.json"
    if resume and complete.exists():
        job = read_json(complete)
    else:
        job = train_job(data_root, split, output, variant == "image_encoder_plus_peak_features", seed,
                        device=device, epochs=epochs, resume=resume)
    provenance = job.get("encoder_provenance", {})
    if provenance.get("loaded_tensor_count") != 120 or not provenance.get("checkpoint_sha256"):
        raise RuntimeError(f"Frozen SimCLR provenance is incomplete for {output}.")
    return job


def run_study(data_root, drive_results_root, run_name, device, seeds, epochs, resume, repository_root=None):
    root = Path(drive_results_root) / run_name
    rows = load_pairwise_rows(data_root)
    protocol = create_fixed_five_fold_unseen_image_splits(rows, root / "protocol" / "five_fold_unseen_image_split.json")
    write_json(root / "protocol" / "completed_task_result.json", {"status": "completed", "source_commit": _source_commit(repository_root),
        "pair_count": len(rows), "decisive_pair_count": sum(row["winner"] in {"1", "2"} for row in rows),
        "protocol": protocol["protocol"], "fold_decisive_counts": [fold["decisive_pair_counts"] for fold in protocol["folds"]]})
    all_jobs, all_evaluations = [], []
    for split in tqdm(protocol["folds"], desc="five unseen-image outer folds", unit="fold"):
        fold_index = split["fold_index"]
        fold_root = root / "outer_image_folds" / f"fold_{fold_index:02d}"
        fold_jobs = []
        for variant in VARIANTS:
            for seed in seeds:
                output = fold_root / "training" / variant / f"seed_{seed:03d}"
                job = _load_or_train(data_root, split, output, variant, seed, device, epochs, resume)
                job["outer_fold"] = fold_index
                fold_jobs.append(job); all_jobs.append(job)
                calibration = fit_validation_calibration(read_json(job["validation_prediction_records"]))
                evaluation_root = fold_root / "sealed_test" / variant / f"seed_{seed:03d}"
                completed = evaluation_root / "completed_task_result.json"
                evaluation = read_json(completed) if resume and completed.exists() else evaluate_sealed_test_set(
                    data_root, split, job["final_model_weights"], evaluation_root,
                    variant == "image_encoder_plus_peak_features", device, calibration, include_embeddings=True)
                all_evaluations.append({"outer_fold": fold_index, "variant": variant, "seed": seed, "evaluation": evaluation})
        write_json(fold_root / "completed_task_result.json", {"status": "completed", "outer_fold": fold_index,
            "test_image_manifest": split["images"]["test"], "pair_counts": split["pair_counts"],
            "decisive_pair_counts": split["decisive_pair_counts"], "training_jobs": fold_jobs})
    validation_scores = {variant: mean([job["validation_pairwise_winner_accuracy"] for job in all_jobs if job["variant"] == variant]) for variant in VARIANTS}
    selected_variant = max(validation_scores, key=lambda variant: validation_scores[variant])
    by_variant_fold = defaultdict(lambda: defaultdict(list))
    for entry in all_evaluations:
        metrics = entry["evaluation"]["metrics"]
        by_variant_fold[entry["variant"]][entry["outer_fold"]].append(metrics)
    aggregate = {}
    for variant in VARIANTS:
        fold_metrics = []
        for fold_index in range(5):
            metrics = by_variant_fold[variant][fold_index]
            fold_metrics.append({"outer_fold": fold_index,
                "decisive_pairwise_accuracy": mean([item["decisive_pairwise_accuracy"] for item in metrics]),
                "all_label_macro_f1": mean([item["macro_f1"] for item in metrics]),
                "brier_score": mean([item["calibration_brier_score_on_decisive_pairs"] for item in metrics]),
                "test_run_count": len(metrics)})
        per_type = {}
        reconstruction_types = sorted({reconstruction_type for entry in all_evaluations if entry["variant"] == variant
                                       for reconstruction_type in entry["evaluation"]["metrics"]["per_reconstruction_type"]})
        for reconstruction_type in reconstruction_types:
            fold_type_accuracy = []
            fold_type_counts = []
            for fold_index in range(5):
                type_metrics = [item["per_reconstruction_type"].get(reconstruction_type, {})
                                for item in by_variant_fold[variant][fold_index]]
                fold_type_accuracy.append(mean([item.get("decisive_accuracy") for item in type_metrics]))
                fold_type_counts.append(mean([item.get("decisive_pair_count") for item in type_metrics]))
            per_type[reconstruction_type] = {"mean_decisive_pair_count_per_fold": mean(fold_type_counts),
                                             "decisive_accuracy": bootstrap_interval(fold_type_accuracy, seed=1000 + len(per_type))}
        aggregate[variant] = {"fold_metrics": fold_metrics,
            "decisive_pairwise_accuracy": bootstrap_interval([item["decisive_pairwise_accuracy"] for item in fold_metrics]),
            "all_label_macro_f1": bootstrap_interval([item["all_label_macro_f1"] for item in fold_metrics], seed=79),
            "brier_score": bootstrap_interval([item["brier_score"] for item in fold_metrics], seed=123),
            "per_reconstruction_type": per_type,
            "test_run_count": sum(item["test_run_count"] for item in fold_metrics)}
    all_images = sorted({image for row in rows for image in (row["left"], row["right"])})
    deployment_split = {"images": {"train": all_images, "validation": [], "test": []},
                        "pair_ids_by_role": {"train": [row["pair_id"] for row in rows], "validation": [], "test": []}}
    deployment_root = root / "final_deployment_model" / selected_variant
    deployment_job = _load_or_train(data_root, deployment_split, deployment_root, selected_variant, 909, device, epochs, resume)
    feature_export = export_pair_selection_features(data_root, deployment_job["final_model_weights"], deployment_root / "active_learning_features",
                                                    selected_variant == "image_encoder_plus_peak_features", device)
    result = {"status": "completed", "run_name": run_name, "source_commit": _source_commit(repository_root),
        "protocol": protocol["protocol"], "seeds": seeds, "epochs": epochs,
        "frozen_simclr_interpretation": protocol["pretraining_policy"],
        "architecture_validation_mean_decisive_accuracy": validation_scores, "selected_deployment_architecture": selected_variant,
        "architecture_test_summary": aggregate, "test_runs_per_architecture": 15,
        "final_deployment_training": deployment_job,
        "active_learning_feature_export": {"record_count": feature_export["record_count"],
            "drive_path": str(deployment_root / "active_learning_features" / "active_learning_pair_selection_features.json"),
            "publication_policy": "The large selector feature matrix remains in Drive; GitHub receives only compact scientific summaries."},
        "next_research_step": "Use the selected frozen-SimCLR reward model to rank unlabeled candidate pairs by calibrated reward-margin entropy; label the high-entropy and embedding-diverse pairs first."}
    write_json(root / "completed_task_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--drive-results-root", required=True)
    parser.add_argument("--run-name", default="five_fold_unseen_image_reward_classifier_t4")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seeds", default="42,79,123")
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--repository-root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_study(args.data_root, args.drive_results_root, args.run_name, args.device,
        [int(seed) for seed in args.seeds.split(",")], args.epochs, args.resume, args.repository_root), indent=2))


if __name__ == "__main__": main()

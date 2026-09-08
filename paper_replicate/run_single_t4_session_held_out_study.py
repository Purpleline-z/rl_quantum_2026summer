"""One-runtime static RHEED study with session-held-out outer evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from .evaluate_image_disjoint_reconstruction_reward_model import evaluate_sealed_test_set, fit_validation_calibration
from .pairwise_and_absolute_label_dataset import (create_session_held_out_split, load_pairwise_rows,
                                                   session_held_out_audit)
from .train_peak_aware_reconstruction_reward_model import train_job


VARIANTS = ("image_encoder_only", "image_encoder_plus_peak_features")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values):
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _selected_variant(jobs):
    by_variant = defaultdict(list)
    for job in jobs:
        by_variant[job["variant"]].append(job["validation_pairwise_winner_accuracy"])
    averages = {variant: mean(scores) for variant, scores in by_variant.items()}
    if any(score is None for score in averages.values()):
        raise ValueError("An outer fold has no decisive validation pairs; add labels before model selection.")
    return max(averages, key=averages.get), averages


def _source_commit(repository_root):
    if not repository_root: return None
    run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root, capture_output=True, text=True, check=True)
    return run.stdout.strip()


def run_study(data_root, drive_results_root, run_name, device, seeds, epochs, resume, repository_root=None):
    run_root = Path(drive_results_root) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    rows = load_pairwise_rows(data_root)
    audit = session_held_out_audit(rows)
    audit.update({"status": "completed", "source_commit": _source_commit(repository_root),
                  "created_at_unix_seconds": time.time(), "pair_count": len(rows)})
    write_json(run_root / "protocol" / "session_held_out_data_readiness_audit.json", audit)
    if not audit["eligible_to_train"]:
        report = {
            "status": "stopped_insufficient_sealed_test_labels", "run_name": run_name,
            "reason": "At least one untouched acquisition session has fewer than 50 decisive pairs.",
            "audit_path": str(run_root / "protocol" / "session_held_out_data_readiness_audit.json"),
            "next_research_step": "Label the reported number of additional decisive comparisons in every under-sized session, then rerun this exact command.",
        }
        write_json(run_root / "completed_task_result.json", report)
        return report

    outer_results = []
    folds = audit["folds"]
    progress = tqdm(folds, desc="outer session folds", unit="fold")
    for fold in progress:
        session = fold["held_out_session"]
        fold_root = run_root / "outer_session_folds" / f"held_out_session_{session}"
        split = create_session_held_out_split(rows, session, fold_root / "protocol" / "sealed_session_split.json")
        jobs = []
        for variant in VARIANTS:
            for seed in seeds:
                output = fold_root / "architecture_comparison" / variant / f"seed_{seed:03d}"
                complete = output / "completed_task_result.json"
                if resume and complete.exists():
                    job = read_json(complete)
                else:
                    job = train_job(data_root, split, output, variant == "image_encoder_plus_peak_features", seed,
                                    device=device, epochs=epochs, resume=resume)
                if job.get("encoder_provenance", {}).get("loaded_tensor_count") != 120:
                    raise RuntimeError(f"{output} did not load the required 120 SimCLR encoder tensors.")
                jobs.append(job)
        selected, validation_means = _selected_variant(jobs)
        chosen_jobs = [job for job in jobs if job["variant"] == selected]
        evaluations = []
        for job in chosen_jobs:
            calibration = fit_validation_calibration(read_json(job["validation_prediction_records"]))
            evaluation_root = fold_root / "sealed_test_evaluations" / selected / f"seed_{job['seed']:03d}"
            completed = evaluation_root / "completed_task_result.json"
            evaluation = read_json(completed) if resume and completed.exists() else evaluate_sealed_test_set(
                data_root, split, job["final_model_weights"], evaluation_root,
                selected == "image_encoder_plus_peak_features", device, calibration, include_embeddings=True)
            evaluations.append({"seed": job["seed"], "evaluation": evaluation})
        summary = {"status": "completed", "held_out_test_session": session, "selected_architecture": selected,
                   "architecture_validation_mean_accuracy": validation_means,
                   "selection_rule": "Highest mean decisive validation accuracy across all five seeds; test-session labels were not used.",
                   "simclr_tensor_count_per_training_job": [job["encoder_provenance"]["loaded_tensor_count"] for job in jobs],
                   "seed_evaluations": evaluations,
                   "mean_sealed_decisive_accuracy": mean([entry["evaluation"]["metrics"]["decisive_pairwise_accuracy"] for entry in evaluations]),
                   "mean_sealed_macro_f1": mean([entry["evaluation"]["metrics"]["macro_f1"] for entry in evaluations])}
        write_json(fold_root / "completed_task_result.json", summary)
        outer_results.append(summary)
        progress.set_postfix(session=session, architecture=selected, test_accuracy=summary["mean_sealed_decisive_accuracy"])
    final = {"status": "completed", "run_name": run_name, "protocol": "three_session_held_out_outer_folds",
             "source_commit": _source_commit(repository_root), "seeds": seeds, "epochs": epochs,
             "outer_fold_results": outer_results,
             "mean_outer_fold_decisive_accuracy": mean([fold["mean_sealed_decisive_accuracy"] for fold in outer_results]),
             "mean_outer_fold_macro_f1": mean([fold["mean_sealed_macro_f1"] for fold in outer_results]),
             "expected_active_learning_output": "Each sealed-fold evaluation contains per-pair SimCLR embedding, five-type reward margin, calibrated preference probability, and entropy.",
             "next_research_step": "Use the selected architecture only if its three-session mean improves both decisive accuracy and macro-F1; otherwise retain image-only encoding and prioritize additional expert pair labels."}
    write_json(run_root / "completed_task_result.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--drive-results-root", required=True)
    parser.add_argument("--run-name", default="session_held_out_static_rheed_classifier_t4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,79,123,202,303")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--repository-root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_study(args.data_root, args.drive_results_root, args.run_name, args.device,
                       [int(seed) for seed in args.seeds.split(",")], args.epochs, args.resume, args.repository_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

"""Drive-friendly queue runner; every completed task has a durable JSON record."""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

from .pairwise_and_absolute_label_dataset import create_image_disjoint_split, load_pairwise_rows
from .train_peak_aware_reconstruction_reward_model import train_job
from .evaluate_image_disjoint_reconstruction_reward_model import evaluate_sealed_test_set
from .export_active_learning_pair_selection_features import export_pair_selection_features

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2), encoding="utf-8")

def split_path(root): return root / "protocol" / "image_disjoint_test_split.json"
def load_split(root): return json.loads(split_path(root).read_text(encoding="utf-8"))

def task_directory(root, task, seed=None, variant=None):
    path = root / task
    if seed is not None: path /= f"seed_{seed:03d}"
    if variant: path /= variant
    return path

def train_comparison(args, root, task_name, seeds, variants):
    split = load_split(root); completed = []
    for seed in seeds:
        for variant in variants:
            directory = task_directory(root, task_name, seed, variant)
            durable = directory / "completed_task_result.json"
            if args.resume and durable.exists(): completed.append(json.loads(durable.read_text())); continue
            completed.append(train_job(args.data_root, split, directory, variant == "image_encoder_plus_peak_features", seed, args.device, resume=args.resume))
    write_json(root / task_name / "completed_task_result.json", {"status": "completed", "jobs": completed})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--drive-results-root", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--model-variants", default="image_encoder_only,image_encoder_plus_peak_features")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-heartbeat-minutes", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); root = Path(args.drive_results_root); root.mkdir(parents=True, exist_ok=True)
    task = args.task_name
    if task == "verify_input_data_and_create_image_disjoint_split":
        rows = load_pairwise_rows(args.data_root); value = create_image_disjoint_split(rows, split_path(root)); write_json(root / "protocol" / "completed_task_result.json", {"status": "completed", "image_count": sum(map(len, value["images"].values()))})
    elif task == "audit_pairwise_absolute_and_ideal_label_coverage":
        rows = load_pairwise_rows(args.data_root); labels = {}; [labels.setdefault(r["reconstruction_type"], 0) for r in rows]
        for r in rows: labels[r["reconstruction_type"]] += 1
        write_json(root / task / "completed_task_result.json", {"status": "completed", "pair_count": len(rows), "pairs_by_reconstruction_type": labels})
    elif task == "validate_peak_feature_extraction_on_training_and_validation_images":
        from .peak_aware_static_rheed_reward_model import RHEEDPeakFeatureExtractor
        from .train_peak_aware_reconstruction_reward_model import TRANSFORM
        from PIL import Image
        split = load_split(root); image = split["images"]["train"][0]; features = RHEEDPeakFeatureExtractor()(TRANSFORM(Image.open(image).convert("RGB")).unsqueeze(0))
        write_json(root / task / "completed_task_result.json", {"status": "completed", "feature_shape": list(features.shape), "finite": bool(features.isfinite().all())})
    elif task == "run_model_and_data_protocol_tests":
        run = subprocess.run([sys.executable, "-m", "unittest", "paper_replicate.tests.test_peak_aware_static_rheed_reward_model"], capture_output=True, text=True)
        write_json(root / task / "completed_task_result.json", {"status": "completed" if run.returncode == 0 else "failed", "stdout": run.stdout, "stderr": run.stderr}); raise SystemExit(run.returncode)
    elif task in {"compare_image_encoder_with_peak_aware_encoder", "independently_repeat_peak_aware_model_comparison"}:
        result_name = "image_encoder_and_peak_feature_comparison" if task == "compare_image_encoder_with_peak_aware_encoder" else "independent_peak_aware_model_repetition"
        train_comparison(args, root, result_name, [int(x) for x in args.seeds.split(",")], args.model_variants.split(","))
    elif task == "train_selected_model_and_evaluate_sealed_image_disjoint_test_set":
        split = load_split(root); comparison = root / "image_encoder_and_peak_feature_comparison" / "completed_task_result.json"
        if not comparison.exists(): raise FileNotFoundError("Start this command after the comparison queue writes its completed_task_result.json.")
        jobs = json.loads(comparison.read_text())["jobs"]
        repeated = root / "independent_peak_aware_model_repetition" / "completed_task_result.json"
        if repeated.exists(): jobs += json.loads(repeated.read_text())["jobs"]
        selected = max(jobs, key=lambda job: (-1 if job.get("validation_pairwise_winner_accuracy") is None else job["validation_pairwise_winner_accuracy"]))
        output = root / task
        final_training = train_job(args.data_root, split, output / "final_selected_model_training", selected["variant"] == "image_encoder_plus_peak_features", 909, args.device, resume=args.resume)
        result = evaluate_sealed_test_set(args.data_root, split, final_training["final_model_weights"], output, selected["variant"] == "image_encoder_plus_peak_features", args.device)
        export_pair_selection_features(args.data_root, final_training["final_model_weights"], output / "active_learning_features", selected["variant"] == "image_encoder_plus_peak_features", args.device)
        write_json(output / "selected_architecture.json", {"selected_variant": selected["variant"], "validation_selection_record": selected, "final_training": final_training, "selection_source": str(comparison), "evaluation": result})
    elif task == "summarize_completed_gpu_tasks_and_prepare_active_learning_features":
        items = list(root.rglob("completed_task_result.json")); write_json(root / task / "completed_task_result.json", {"status": "completed", "completed_task_json_paths": [str(p) for p in items]})
    else: raise ValueError(f"Unknown task name: {task}")

if __name__ == "__main__": main()

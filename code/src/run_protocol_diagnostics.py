#!/usr/bin/env python3
"""Manifest-scoped diagnostic studies for the current RHEED classifier.

The runner deliberately separates inner utility-validation calibration from the
outer ideal-image confirmation.  It reuses frozen Stage-1 selected-pair IDs;
therefore none of these studies reruns active selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from active_learning_pipeline import Config, Experiment, PairRows, canonical_pair, seed_everything
from project_paths import CODE_ROOT, RESULT_ROOT


STUDIES = RESULT_ROOT / "protocol_diagnostics"
STAGE1 = RESULT_ROOT / "selection_benchmark" / "stage1_selector_curves_none"
CALIBRATION_SEEDS = (42, 79, 123)
CONFIRMATION_SEEDS = (42, 79, 123, 202, 303)
SELECTORS = ("random", "uncertainty")
LRS = (1e-5, 3e-5, 1e-4, 3e-4)


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_path(study_id: str) -> Path:
    if not study_id.replace("_", "").isalnum():
        raise ValueError("Study IDs may contain only letters, numbers, and underscores.")
    return STUDIES / study_id


def stage1_job(seed: int, selector: str) -> Path:
    return STAGE1 / "jobs" / f"seed-{seed}_budget-100_strategy-{selector}_symmetry-none"


def frozen_selection(seed: int, selector: str) -> tuple[list[str], Path]:
    path = stage1_job(seed, selector) / "manifests" / "selected_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed Stage-1 selection: {path}")
    frame = pd.read_csv(path)
    if "pair_id" not in frame:
        raise ValueError(f"Stage-1 selected-pair artifact has no pair_id column: {path}")
    ids = list(dict.fromkeys(frame.pair_id.astype(str)))
    if len(ids) != 100:
        raise ValueError(f"Expected 100 frozen selected pair groups in {path}; found {len(ids)}.")
    return ids, path


def build_manifest(study_id: str, kind: str, jobs: list[dict], protocol: dict) -> Path:
    root = study_path(study_id)
    payload = {"study_id": study_id, "kind": kind, "protocol": protocol, "jobs": jobs}
    path = root / "study_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise FileExistsError(f"Immutable manifest already exists with different content: {path}")
    atomic_json(payload, path)
    return path


def generate_lr_calibration() -> Path:
    jobs = [{"job_id": f"seed-{seed}_selector-{selector}_lr-{lr:.0e}", "seed": seed,
             "selector": selector, "lr": lr, "phase": "calibration"}
            for seed in CALIBRATION_SEEDS for selector in SELECTORS for lr in LRS]
    return build_manifest("learning_rate_calibration", "learning_rate_calibration", jobs,
                          {"epochs": 3, "learning_rates": LRS, "seeds": CALIBRATION_SEEDS,
                           "selectors": SELECTORS, "split_used_for_selection": "utility_validation",
                           "outer_test_access_during_calibration": False, "selection_rule":
                           "highest mean utility-validation accuracy; then lower sample SD; then lower learning rate"})


def generate_encoder_screen() -> Path:
    jobs = [{"job_id": f"seed-{seed}_selector-{selector}_init-{init}", "seed": seed,
             "selector": selector, "encoder_initialization": init, "phase": "screening"}
            for seed in CALIBRATION_SEEDS for selector in SELECTORS for init in ("simclr", "imagenet")]
    return build_manifest("encoder_initialization_screen", "encoder_initialization_screen", jobs,
                          {"epochs": 3, "learning_rate": 1e-4, "seeds": CALIBRATION_SEEDS,
                           "selectors": SELECTORS, "initializations": ("simclr", "imagenet"),
                           "split_used_for_selection": "utility_validation",
                           "outer_test_access_during_screening": False})


def _experiment(spec: dict, data_root: str | None) -> tuple[Experiment, list[str], list[str], Path]:
    cfg = Config(initial_pairs=50, candidate_pairs=120, budget=100, epochs=3,
                 lr=float(spec.get("lr", 1e-4)), seed=int(spec["seed"]), strategies=spec["selector"],
                 acquisition_mode="single-shot", utility_per_pair=False, symmetry_mode="none",
                 data_root=data_root, encoder_initialization=spec.get("encoder_initialization", "simclr"))
    exp = Experiment(cfg)
    initial, pool = exp.load_and_split()
    selected, source = frozen_selection(cfg.seed, spec["selector"])
    if not set(selected) <= set(pool) or set(selected) & set(initial):
        raise ValueError("Frozen Stage-1 selected IDs are incompatible with the reconstructed initial/pool split.")
    return exp, initial, selected, source


def _run_current(spec: dict, directory: Path, data_root: str | None) -> None:
    exp, initial, selected, source = _experiment(spec, data_root)
    model, train_metrics = exp.train(initial + selected)
    utility = exp.evaluate(model, split="utility_validation")
    # Calibration/screening never touches the outer test.  Confirmation does.
    outer = exp.evaluate(model, split="outer_test") if spec["phase"] == "confirmation" else None
    audit = exp.protocol_audit(initial, exp.candidate_metadata.keys())
    atomic_json({"spec": spec, "config": asdict(exp.cfg), "frozen_selection_source": str(source),
                 "frozen_selection_sha256": sha256(source), "selected_pair_ids": selected,
                 "train_metrics": train_metrics, "utility_validation": utility, "outer_test": outer,
                 "audit": audit, "checkpointing_enabled": False}, directory / "result.json")


def run_manifest(manifest_path: Path, data_root: str | None = None) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for spec in payload["jobs"]:
        directory = manifest_path.parent / "jobs" / spec["job_id"]
        if (directory / "result.json").exists():
            print(f"{spec['job_id']}: completed")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        _run_current(spec, directory, data_root)
        print(f"{spec['job_id']}: completed")


def _results(manifest_path: Path) -> pd.DataFrame:
    payload = json.loads(manifest_path.read_text(encoding="utf-8")); rows = []
    for spec in payload["jobs"]:
        path = manifest_path.parent / "jobs" / spec["job_id"] / "result.json"
        if not path.exists():
            raise ValueError(f"Study is incomplete; missing {spec['job_id']}")
        result = json.loads(path.read_text(encoding="utf-8")); utility = result["utility_validation"]
        rows.append({**spec, "utility_validation_accuracy": utility["test_accuracy"],
                     "utility_validation_correct": utility["test_correct"], "utility_validation_total": utility["test_total"],
                     "source_result": str(path), "source_sha256": sha256(path)})
    return pd.DataFrame(rows)


def _choose(frame: pd.DataFrame, key: str) -> str:
    summary = frame.groupby(key, as_index=False).agg(mean=("utility_validation_accuracy", "mean"),
                                                      std=("utility_validation_accuracy", "std"))
    # Lowest numeric learning rate is the final deterministic tiebreaker; for
    # encoder names, alphabetical order makes the rule reproducible.
    summary["tie_key"] = summary[key].map(lambda x: float(x) if isinstance(x, (int, float)) else str(x))
    return str(summary.sort_values(["mean", "std", "tie_key"], ascending=[False, True, True]).iloc[0][key])


def _plot_summary(summary: pd.DataFrame, x: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for selector, group in summary.groupby("selector"):
        ax.errorbar(group[x].astype(str), group["mean"], yerr=group["std"], marker="o", capsize=3, label=selector)
    ax.set(title=title, xlabel=x, ylabel="Utility-validation ideal-image accuracy"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def aggregate_lr_calibration(manifest_path: Path) -> Path:
    frame = _results(manifest_path); output = manifest_path.parent / "aggregate"; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_utility_validation.csv", index=False)
    summary = frame.groupby(["selector", "lr"], as_index=False).agg(mean=("utility_validation_accuracy", "mean"), std=("utility_validation_accuracy", "std"), n=("seed", "count"))
    summary.to_csv(output / "learning_rate_utility_validation_summary.csv", index=False)
    chosen = float(_choose(frame, "lr")); atomic_json({"locked_learning_rate": chosen, "selection_split": "utility_validation", "outer_test_used": False,
        "selection_rule": "highest mean; then lower sample SD; then lower learning rate"}, output / "locked_hyperparameter.json")
    _plot_summary(summary, "lr", "Learning-rate calibration (outer test withheld)", output / "utility_validation_by_learning_rate.png")
    return output


def generate_lr_confirmation(manifest_path: Path) -> Path:
    locked = json.loads((manifest_path.parent / "aggregate" / "locked_hyperparameter.json").read_text(encoding="utf-8"))["locked_learning_rate"]
    jobs = [{"job_id": f"seed-{seed}_selector-{selector}_lr-{locked:.0e}_confirmation", "seed": seed, "selector": selector,
             "lr": locked, "phase": "confirmation"} for seed in CONFIRMATION_SEEDS for selector in SELECTORS]
    return build_manifest("learning_rate_confirmation", "learning_rate_confirmation", jobs,
        {"epochs": 3, "locked_learning_rate": locked, "seeds": CONFIRMATION_SEEDS, "selectors": SELECTORS,
         "selection_source": str(manifest_path.parent / "aggregate" / "locked_hyperparameter.json")})


def aggregate_confirmation(manifest_path: Path, label: str) -> Path:
    frame = _results(manifest_path); output = manifest_path.parent / "aggregate"; output.mkdir(parents=True, exist_ok=True)
    outer = []
    for result_path in frame.source_result:
        result = json.loads(Path(result_path).read_text(encoding="utf-8")); outer.append(result["outer_test"])
    frame["outer_test_accuracy"] = [x["test_accuracy"] for x in outer]
    frame.to_csv(output / "per_seed_confirmation.csv", index=False)
    summary = frame.groupby("selector", as_index=False).agg(n=("seed", "count"), outer_test_accuracy_mean=("outer_test_accuracy", "mean"), outer_test_accuracy_std=("outer_test_accuracy", "std"))
    summary.to_csv(output / "outer_test_confirmation_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(summary.selector, summary.outer_test_accuracy_mean, yerr=summary.outer_test_accuracy_std, capsize=4)
    ax.set(title=label, ylabel="Outer ideal-test accuracy"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(output / "outer_test_confirmation.png", dpi=180); plt.close(fig)
    return output


def aggregate_encoder_screen(manifest_path: Path) -> Path:
    frame = _results(manifest_path); output = manifest_path.parent / "aggregate"; output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_utility_validation.csv", index=False)
    summary = frame.groupby(["selector", "encoder_initialization"], as_index=False).agg(mean=("utility_validation_accuracy", "mean"), std=("utility_validation_accuracy", "std"), n=("seed", "count"))
    summary.to_csv(output / "encoder_utility_validation_summary.csv", index=False)
    chosen = _choose(frame, "encoder_initialization"); atomic_json({"locked_encoder_initialization": chosen, "selection_split": "utility_validation", "outer_test_used": False}, output / "locked_encoder.json")
    _plot_summary(summary.rename(columns={"encoder_initialization": "initialization"}), "initialization", "Encoder-initialization screening (outer test withheld)", output / "utility_validation_by_encoder.png")
    return output


def generate_encoder_confirmation(manifest_path: Path) -> Path:
    locked = json.loads((manifest_path.parent / "aggregate" / "locked_encoder.json").read_text(encoding="utf-8"))["locked_encoder_initialization"]
    jobs = [{"job_id": f"seed-{seed}_selector-{selector}_init-{locked}_confirmation", "seed": seed, "selector": selector,
             "encoder_initialization": locked, "phase": "confirmation"} for seed in CONFIRMATION_SEEDS for selector in SELECTORS]
    return build_manifest("encoder_initialization_confirmation", "encoder_initialization_confirmation", jobs,
        {"epochs": 3, "learning_rate": 1e-4, "locked_encoder_initialization": locked, "seeds": CONFIRMATION_SEEDS, "selectors": SELECTORS})


def generate_classifier2_audit(data_root: str | None = None) -> Path:
    """Create a no-training compatibility manifest for the historical bridge."""
    cfg = Config(initial_pairs=50, candidate_pairs=120, seed=42, data_root=data_root)
    exp = Experiment(cfg); exp.load_and_split()
    legacy = CODE_ROOT / "classifier2" / "downstream_classifier_scripts" / "train_unified.py"
    rows = exp.rows_for(exp.groups.keys())
    raw_types = sorted(rows.Reconstruction_Type.astype(str).unique())
    payload = {"study_id": "classifier2_protocol_bridge", "status": "ready_for_pairwise_bridge",
               "current_pairwise_csv": str(exp.pairwise_csv), "current_pairwise_sha256": sha256(exp.pairwise_csv),
               "legacy_training_source": str(legacy), "legacy_training_source_sha256": sha256(legacy),
               "shared_pair_groups": len(exp.groups), "shared_valid_rows": len(rows), "raw_reconstruction_types": raw_types,
               "legacy_protocol": {"pair_level_holdout_fraction": .2, "epochs": 30, "learning_rate": 1e-4,
                                   "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "evaluation": "pairwise_holdout"},
               "current_protocol": {"epochs": 3, "learning_rate": 1e-4, "evaluation": "ideal_image_outer_test"},
               "limitations": ["The legacy report's pairwise holdout and current ideal-image outer test are separate metrics.",
                               "This audit establishes dataset compatibility before any training comparison."]}
    root = study_path("classifier2_protocol_bridge"); atomic_json(payload, root / "compatibility_manifest.json")
    (root / "compatibility_audit.md").write_text("# Classifier2 protocol bridge\n\nThe current v1.8 CSV and the preserved Classifier2 training source are available. The planned bridge uses a fixed pair-level 80/20 holdout and reports pairwise accuracy separately from ideal-image accuracy. It must not call the historical projected 88–92% value a measured result.\n", encoding="utf-8")
    return root / "compatibility_manifest.json"


def run_classifier2_bridge(data_root: str | None = None, seed: int = 42) -> Path:
    """Run a checkpoint-free pairwise bridge on one frozen, shared holdout.

    The preserved legacy script's full ``UnifiedDataset`` also mixes ideal and
    bad anchors and writes checkpoints.  This bridge uses its public pairwise
    model/loss/dataset implementation without those writes, and labels that
    distinction in its output rather than claiming a bit-for-bit recreation.
    """
    root = study_path("classifier2_protocol_bridge")
    manifest = root / "compatibility_manifest.json"
    if not manifest.exists():
        generate_classifier2_audit(data_root)
    cfg = Config(initial_pairs=50, candidate_pairs=120, seed=seed, data_root=data_root, epochs=3, lr=1e-4)
    exp = Experiment(cfg); exp.load_and_split()
    ids = list(exp.groups); random.Random(seed).shuffle(ids); split = int(len(ids) * .8)
    train_ids, test_ids = ids[:split], ids[split:]
    legacy_root = CODE_ROOT / "classifier2" / "downstream_classifier_scripts"
    sys.path.insert(0, str(legacy_root))
    from bradley_terry_model import BradleyTerryRewardModel, PairwiseDataset, train_epoch, evaluate as legacy_evaluate
    legacy_frame = exp.rows_for(train_ids).copy(); test_frame = exp.rows_for(test_ids).copy()
    legacy_frame["Winner"] = legacy_frame.Winner.astype(str).replace({"1": "image1", "2": "image2"})
    test_frame["Winner"] = test_frame.Winner.astype(str).replace({"1": "image1", "2": "image2"})
    # Preserve only types understood by the legacy public model.
    supported = {"(1 x 1)", "Twinned(2 x 1)", "c(6 x 2)", "HTR"}
    legacy_frame = legacy_frame[legacy_frame.Reconstruction_Type.isin(supported)].copy()
    test_frame = test_frame[test_frame.Reconstruction_Type.isin(supported)].copy()
    device = exp.device; seed_everything(seed)
    legacy_model = BradleyTerryRewardModel(pretrained_encoder_path=None).to(device)
    state = torch.load(exp.encoder_weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    cleaned = {k.replace("encoder.", ""): v for k, v in state.items() if not k.startswith("projector.")}
    missing, unexpected = legacy_model.encoder.encoder.load_state_dict(cleaned, strict=False)
    train_ds = PairwiseDataset(legacy_frame, exp.data_root); test_ds = PairwiseDataset(test_frame, exp.data_root)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(legacy_model.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(30): train_epoch(legacy_model, loader, optimizer, device)
    legacy_metrics = legacy_evaluate(legacy_model, DataLoader(test_ds, batch_size=32), device)
    current_model, current_train = exp.train(train_ids)
    current_metrics = {"accuracy": exp.pairwise_accuracy(current_model, exp.rows_for(test_ids)),
                       "denominator_rows": int(len(exp.rows_for(test_ids)))}
    rows = pd.DataFrame([
        {"system": "Classifier2 pairwise-only bridge", "pairwise_accuracy": legacy_metrics["accuracy"], "correct": legacy_metrics["correct"], "total": legacy_metrics["total"], "epochs": 30},
        {"system": "Current pipeline", "pairwise_accuracy": current_metrics["accuracy"], "correct": None, "total": current_metrics["denominator_rows"], "epochs": 3},
    ])
    rows.to_csv(root / "pairwise_bridge_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(rows.system, rows.pairwise_accuracy); ax.set_ylim(0, 1); ax.set_ylabel("Pairwise holdout accuracy"); ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(root / "pairwise_bridge_comparison.png", dpi=180); plt.close(fig)
    atomic_json({"seed": seed, "train_pair_groups": len(train_ids), "test_pair_groups": len(test_ids),
                 "legacy_checkpoint_load": {"missing": list(missing), "unexpected": list(unexpected)},
                 "legacy_protocol_deviation": "pairwise-only bridge; no UnifiedDataset anchors and no checkpoint files",
                 "ideal_image_comparison": "not performed: legacy and current ideal-image split/evaluation contracts are not identical",
                 "current_train_metrics": current_train}, root / "bridge_run_manifest.json")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate-lr-calibration", "generate-encoder-screen", "generate-classifier2-audit"):
        item = sub.add_parser(name); item.add_argument("--data-root")
    run = sub.add_parser("run-manifest"); run.add_argument("--manifest", type=Path, required=True); run.add_argument("--data-root")
    bridge = sub.add_parser("run-classifier2-bridge"); bridge.add_argument("--data-root"); bridge.add_argument("--seed", type=int, default=42)
    for name in ("aggregate-lr-calibration", "aggregate-encoder-screen", "generate-lr-confirmation", "generate-encoder-confirmation"):
        item = sub.add_parser(name); item.add_argument("--manifest", type=Path, required=True)
    for name in ("aggregate-lr-confirmation", "aggregate-encoder-confirmation"):
        item = sub.add_parser(name); item.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-lr-calibration": print(generate_lr_calibration())
    elif args.command == "generate-encoder-screen": print(generate_encoder_screen())
    elif args.command == "generate-classifier2-audit": print(generate_classifier2_audit(args.data_root))
    elif args.command == "run-classifier2-bridge": print(run_classifier2_bridge(args.data_root, args.seed))
    elif args.command == "run-manifest": run_manifest(args.manifest, args.data_root)
    elif args.command == "aggregate-lr-calibration": print(aggregate_lr_calibration(args.manifest))
    elif args.command == "aggregate-encoder-screen": print(aggregate_encoder_screen(args.manifest))
    elif args.command == "generate-lr-confirmation": print(generate_lr_confirmation(args.manifest))
    elif args.command == "generate-encoder-confirmation": print(generate_encoder_confirmation(args.manifest))
    elif args.command == "aggregate-lr-confirmation": print(aggregate_confirmation(args.manifest, "Learning-rate outer-test confirmation"))
    else: print(aggregate_confirmation(args.manifest, "Encoder initialization outer-test confirmation"))


if __name__ == "__main__":
    main()

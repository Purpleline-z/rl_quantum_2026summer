#!/usr/bin/env python3
"""Run the resumable one-T4 simulator-to-real preliminary queue.

The queue has 15 cells: five real-only baselines, five synthetic-only transfer
diagnostics, and five synthetic-pretrain→real-fine-tune treatments.  It saves
one checkpoint after every epoch to Drive and removes it after a cell finishes.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from simulator_study_common import (
    CODE_ROOT, PROTOCOL, FiveOutputClassifier2CompatibleModel, RealUnifiedDataset,
    SyntheticDataset, atomic_json_write, evaluate_three_class, extract_synthetic_archive,
    load_partition_table, load_real_training_inputs, progress, real_collate,
    resolve_data_root, save_checkpoint, sha256_file, synthetic_rows, validate_synthetic_archive,
    pairwise_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def result_path(output: Path, arm: str, seed: int) -> Path:
    return output / "per_seed_results" / arm / f"seed_{seed}_completed_result.json"


def checkpoint_path(output: Path, arm: str, seed: int) -> Path:
    return output / "resumable_checkpoints" / arm / f"seed_{seed}_resume_checkpoint.pth"


def verify_preflight(output: Path, archive: Path) -> None:
    required = [output / "preflight_audits" / "synthetic_archive_validation.json", output / "preflight_audits" / "strict_real_image_partition_audit.json", output / "preflight_audits" / "real_pairwise_capacity_audit.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise RuntimeError("GPU queue requires completed CPU preflight files: " + "; ".join(missing))
    report = json.loads(required[0].read_text())
    if report.get("status") != "pass" or report.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError("Synthetic archive does not match the validated preflight archive.")


def model(device: torch.device) -> FiveOutputClassifier2CompatibleModel:
    checkpoint = CODE_ROOT / "classifier2" / "simclr_encoder_checkpoint" / "simclr_resnet18_encoder.pth"
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    return FiveOutputClassifier2CompatibleModel(checkpoint).to(device)


def restore(path: Path, current_model, optimizer, expected_phase: str) -> int:
    if not path.is_file(): return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["phase"] != expected_phase: return 0
    current_model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
    progress(f"Resuming {expected_phase} at epoch {state['completed_epochs']}")
    return int(state["completed_epochs"])


def save_phase(path: Path, current_model, optimizer, phase: str, completed_epochs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save({"phase": phase, "completed_epochs": completed_epochs, "model": current_model.state_dict(), "optimizer": optimizer.state_dict()}, temporary)
    temporary.replace(path)


def train_synthetic(current_model, synthetic_root: Path, seed: int, device: torch.device, checkpoint: Path, phase: str = "synthetic") -> dict:
    train_rows, validation_rows = synthetic_rows(synthetic_root, seed)
    cfg = PROTOCOL["synthetic_pretraining"]
    loader = DataLoader(SyntheticDataset(train_rows, synthetic_root), batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(current_model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    start = restore(checkpoint, current_model, optimizer, phase)
    validation_loader = DataLoader(SyntheticDataset(validation_rows, synthetic_root), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    for epoch in range(start, cfg["epochs"]):
        current_model.train(); losses = []
        for images, labels in loader:
            scores = current_model(images.to(device)); selected = scores[:, [1, 2, 3]]
            mapped = torch.tensor([{1: 0, 2: 1, 3: 2}[int(value)] for value in labels], device=device)
            loss = F.cross_entropy(selected, mapped); optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        current_model.eval(); correct = total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                pred = current_model(images.to(device))[:, [1, 2, 3]].argmax(dim=1).cpu().numpy()
                truth = np.asarray([{1: 0, 2: 1, 3: 2}[int(value)] for value in labels])
                correct += int((pred == truth).sum()); total += len(truth)
        save_phase(checkpoint, current_model, optimizer, phase, epoch + 1)
        progress(f"{phase} seed={seed} epoch={epoch + 1}/{cfg['epochs']} loss={np.mean(losses):.4f} synthetic_validation_accuracy={correct / total:.4f} checkpoint={checkpoint}")
    return {"synthetic_train_images": len(train_rows), "synthetic_validation_images": len(validation_rows), "synthetic_validation_groups": int(validation_rows.same_surface_group.nunique())}


def train_real(current_model, data_root: Path, partition, seed: int, device: torch.device, checkpoint: Path, phase: str = "real") -> dict:
    pairs, anchors, bad, trajectories = load_real_training_inputs(data_root, partition)
    cfg = PROTOCOL["real_training"]
    dataset = RealUnifiedDataset(pairs, anchors, bad, trajectories, cfg["samples_per_epoch"], seed)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, collate_fn=real_collate)
    optimizer = torch.optim.AdamW(current_model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    start = restore(checkpoint, current_model, optimizer, phase)
    for epoch in range(start, cfg["epochs"]):
        current_model.train(); losses = []
        for batch in loader:
            loss = pairwise_loss(current_model, batch, device); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.item()))
        validation = evaluate_three_class(current_model, partition, device, "validation")
        save_phase(checkpoint, current_model, optimizer, phase, epoch + 1)
        progress(f"{phase} seed={seed} epoch={epoch + 1}/{cfg['epochs']} loss={np.mean(losses):.4f} validation_macro_recall={validation['validation_macro_recall']:.4f} checkpoint={checkpoint}")
    return {"real_pairwise_training_rows": len(pairs), "real_training_anchors": {name: len(paths) for name, paths in anchors.items()}, "real_bad_training_images": len(bad)}


def finish(output: Path, arm: str, seed: int, payload: dict, checkpoint: Path) -> None:
    atomic_json_write(payload, result_path(output, arm, seed))
    if checkpoint.exists(): checkpoint.unlink()
    progress(f"Completed {arm} seed={seed}; result={result_path(output, arm, seed)}; deleted_checkpoint={checkpoint}")


def run_cell(arm: str, seed: int, archive: Path, output: Path, device: torch.device, data_root: Path, synthetic_root: Path) -> None:
    destination = result_path(output, arm, seed)
    if destination.is_file():
        progress(f"Skipping completed {arm} seed={seed}: {destination}"); return
    seed_everything(seed); partition = load_partition_table(output); checkpoint = checkpoint_path(output, arm, seed); current = model(device)
    common = {"arm": arm, "seed": seed, "protocol": PROTOCOL, "device": str(device), "synthetic_archive_sha256": sha256_file(archive)}
    if arm == "real_only_classifier2_compatible_baseline":
        common.update(train_real(current, data_root, partition, seed, device, checkpoint))
    elif arm == "synthetic_only_real_transfer_diagnostic":
        common.update(train_synthetic(current, synthetic_root, seed, device, checkpoint))
    elif arm == "synthetic_pretraining_then_real_finetuning":
        # One cell intentionally contains both stages. A single resumable checkpoint
        # is overwritten every epoch and deleted only after the downstream result.
        common.update(train_synthetic(current, synthetic_root, seed, device, checkpoint, "synthetic_before_real"))
        if checkpoint.exists(): checkpoint.unlink()
        common.update(train_real(current, data_root, partition, seed, device, checkpoint, "real_after_synthetic"))
    else: raise ValueError(arm)
    common.update(evaluate_three_class(current, partition, device, "outer_test"))
    finish(output, arm, seed, common, checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run_automatic_queue"])
    parser.add_argument("--synthetic-archive", required=True, type=Path)
    parser.add_argument("--drive-output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root")
    parser.add_argument("--ephemeral-cache", type=Path, default=Path("/content/rheed_simulator_p656_ephemeral_cache"))
    args = parser.parse_args(); archive=args.synthetic_archive.resolve(); output=args.drive_output.resolve(); device=torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; do not silently fall back to CPU.")
    verify_preflight(output, archive); synthetic_root=extract_synthetic_archive(archive, args.ephemeral_cache); data_root=resolve_data_root(args.data_root)
    arms = ["real_only_classifier2_compatible_baseline", "synthetic_only_real_transfer_diagnostic", "synthetic_pretraining_then_real_finetuning"]
    for seed in PROTOCOL["paired_training_seeds"]:
        for arm in arms: run_cell(arm, seed, archive, output, device, data_root, synthetic_root)
    progress("Automatic queue completed.")
if __name__ == "__main__": main()

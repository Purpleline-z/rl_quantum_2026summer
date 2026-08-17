#!/usr/bin/env python3
"""Run the approved, literature-backed strict image-disjoint budget curve.

This runner never uses the earlier custom selectors.  All result paths are
explicitly supplied with --drive-output so Colab work is durable outside
``/content``.  It deliberately separates initial-labelled images, candidate
images, validation images, and outer-test images before pair construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
PROGRAM_ROOT = CODE_ROOT / "active_learning_program"
sys.path.insert(0, str(PROGRAM_ROOT))

from monte_carlo_dropout_uncertainty import score_mc_dropout, select_mc_dropout
from pair_acquisition_methods import build_embedding_cache, random_sampling, uncertainty_sampling
from pair_acquisition_shared_calculations import core_set_select
from pairwise_active_learning_pipeline import Config, Experiment, TYPE_TO_INDEX, canonical_pair, canonical_type
from pairwise_csv_loading_and_image_partitioning import locate_input
from resumable_model_training import train_with_epoch_checkpoints


APPROVED_SELECTORS = (
    "random",
    "predictive_entropy",
    "core_set_k_center",
    "mc_dropout_mutual_information",
    "mc_dropout_probability_variance",
)


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def stable_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def print_progress(label: str, completed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    remaining = elapsed * (total - completed) / completed if completed else float("nan")
    eta = "estimating" if not np.isfinite(remaining) else f"{remaining / 60:.1f} min"
    print(f"[{label}] {completed}/{total} ({100 * completed / total:.1f}%) | elapsed={elapsed / 60:.1f} min | ETA={eta}", flush=True)


class FourWayImageDisjointExperiment(Experiment):
    """An Experiment whose four pairwise image populations cannot overlap."""

    def __init__(self, cfg: Config, protocol: dict[str, Any], output: Path):
        super().__init__(cfg)
        self.protocol, self.output = protocol, output
        self.validation_rows = pd.DataFrame()
        self.holdout_rows = pd.DataFrame()
        self.initial_ids: list[str] = []
        self.candidate_ids: list[str] = []
        self.partition_images: dict[str, set[str]] = {}

    def load_and_split(self) -> tuple[list[str], list[str]]:
        self.pairwise_csv = locate_input(self.data_root.parent, CODE_ROOT, self.dataset.pairwise_name)
        frame = pd.read_csv(self.pairwise_csv)
        needed = {"Image1_Path", "Image2_Path", "Reconstruction_Type", "Winner"}
        if not needed <= set(frame):
            raise ValueError(f"Pairwise CSV lacks columns: {sorted(needed - set(frame))}")
        frame["canonical_type"] = frame.Reconstruction_Type.map(canonical_type)
        frame = frame[frame.canonical_type.notna() & frame.Winner.astype(str).isin(["1", "2", "tie", "not_apply"])].copy()
        frame["resolved_img1"] = [str((self.data_root / p).resolve()) for p in frame.Image1_Path]
        frame["resolved_img2"] = [str((self.data_root / p).resolve()) for p in frame.Image2_Path]
        frame = frame[frame.resolved_img1.map(lambda value: Path(value).is_file()) & frame.resolved_img2.map(lambda value: Path(value).is_file())].copy()
        frame["pair_id"] = [canonical_pair(a, b) for a, b in zip(frame.resolved_img1, frame.resolved_img2)]
        frame["type_idx"] = frame.canonical_type.map(TYPE_TO_INDEX)
        frame["confidence_weight"] = frame.get("Confidence", pd.Series(index=frame.index)).map({"Confident": 1.0, "Somewhat sure": 0.7}).fillna(1.0)
        groups = {pair_id: group.reset_index(drop=True) for pair_id, group in frame.groupby("pair_id", sort=True)}

        all_images = sorted({*frame.resolved_img1, *frame.resolved_img2})
        rng = random.Random(self.cfg.seed)
        rng.shuffle(all_images)
        fractions = self.protocol["four_way_image_split"]
        names = ("initial", "candidate", "validation", "outer_test")
        if set(fractions) != set(names) or not np.isclose(sum(fractions.values()), 1.0):
            raise ValueError("four_way_image_split must contain initial, candidate, validation, outer_test and sum to 1.")
        cut_initial = int(len(all_images) * fractions["initial"])
        cut_candidate = cut_initial + int(len(all_images) * fractions["candidate"])
        cut_validation = cut_candidate + int(len(all_images) * fractions["validation"])
        self.partition_images = {
            "initial": set(all_images[:cut_initial]),
            "candidate": set(all_images[cut_initial:cut_candidate]),
            "validation": set(all_images[cut_candidate:cut_validation]),
            "outer_test": set(all_images[cut_validation:]),
        }

        def membership(group: pd.DataFrame) -> str:
            pair_images = set(group.resolved_img1) | set(group.resolved_img2)
            matches = [name for name, values in self.partition_images.items() if pair_images <= values]
            return matches[0] if len(matches) == 1 else "cross_partition_or_unused"

        memberships = {pair_id: membership(group) for pair_id, group in groups.items()}
        self.initial_ids = sorted(pair_id for pair_id, part in memberships.items() if part == "initial")
        self.candidate_ids = sorted(pair_id for pair_id, part in memberships.items() if part == "candidate")
        self.validation_rows = pd.concat([groups[pair_id] for pair_id, part in memberships.items() if part == "validation"], ignore_index=True) if "validation" in memberships.values() else pd.DataFrame()
        self.holdout_rows = pd.concat([groups[pair_id] for pair_id, part in memberships.items() if part == "outer_test"], ignore_index=True) if "outer_test" in memberships.values() else pd.DataFrame()
        self.groups = {pair_id: groups[pair_id] for pair_id in self.initial_ids + self.candidate_ids}
        self.candidate_metadata = {pair_id: {"pair_id": pair_id, "img1": self.groups[pair_id].iloc[0].resolved_img1, "img2": self.groups[pair_id].iloc[0].resolved_img2} for pair_id in self.candidate_ids}
        self._split_ideals()
        self._load_bad_references()
        self._write_audit(memberships)
        required_initial, required_candidate = int(self.protocol["initial_pair_groups"]), int(self.protocol["maximum_acquired_pair_groups"])
        if len(self.initial_ids) < required_initial or len(self.candidate_ids) < required_candidate:
            raise RuntimeError(f"Strict four-way capacity failed for seed {self.cfg.seed}: initial={len(self.initial_ids)} (need {required_initial}), candidate={len(self.candidate_ids)} (need {required_candidate}).")
        return self.initial_ids[:required_initial], self.candidate_ids[:required_candidate]

    def _write_audit(self, memberships: dict[str, str]) -> None:
        overlaps = {f"{left}__{right}": sorted(self.partition_images[left] & self.partition_images[right]) for index, left in enumerate(self.partition_images) for right in list(self.partition_images)[index + 1:]}
        initial_images = set().union(*(set(self.groups[p].resolved_img1) | set(self.groups[p].resolved_img2) for p in self.initial_ids)) if self.initial_ids else set()
        candidate_images = set().union(*(set(self.groups[p].resolved_img1) | set(self.groups[p].resolved_img2) for p in self.candidate_ids)) if self.candidate_ids else set()
        audit = {
            "seed": self.cfg.seed,
            "image_ids": {name: sorted(values) for name, values in self.partition_images.items()},
            "pair_ids": {name: sorted(pair_id for pair_id, part in memberships.items() if part == name) for name in (*self.partition_images, "cross_partition_or_unused")},
            "pair_rows": {"initial": int(sum(len(self.groups[p]) for p in self.initial_ids)), "candidate": int(sum(len(self.groups[p]) for p in self.candidate_ids)), "validation": len(self.validation_rows), "outer_test": len(self.holdout_rows)},
            "image_overlap_checks": overlaps,
            "initial_candidate_image_overlap": sorted(initial_images & candidate_images),
            "candidate_labels_hidden_from_selector": True,
            "capacity": {"initial_pair_groups": len(self.initial_ids), "candidate_pair_groups": len(self.candidate_ids)},
        }
        if any(overlaps.values()) or audit["initial_candidate_image_overlap"]:
            raise RuntimeError("Four-way image-disjoint leakage audit failed.")
        atomic_json(audit, self.output / "audit" / f"seed_{self.cfg.seed}_four_way_image_partition_audit.json")

    def validation_accuracy(self, model: torch.nn.Module) -> float:
        return self.evaluate(model, "utility_validation")["test_accuracy"]


def make_experiment(protocol: dict[str, Any], output: Path, seed: int, encoder: str, learning_rate: float, epochs: int, weight_decay: float, device: str) -> FourWayImageDisjointExperiment:
    cfg = Config(seed=seed, dataset_version=protocol["dataset_version"], initial_pairs=protocol["initial_pair_groups"], candidate_pairs=protocol["maximum_acquired_pair_groups"], epochs=epochs, lr=learning_rate, weight_decay=weight_decay, train_batch_size=protocol["train_batch_size"], dropout_p=protocol["dropout_probability"], encoder_initialization=encoder, acquisition_mode="single-shot", data_root=None, device=device)
    return FourWayImageDisjointExperiment(cfg, protocol, output)


def select_rankings(exp: FourWayImageDisjointExperiment, candidate_ids: list[str], baseline: torch.nn.Module) -> dict[str, list[str]]:
    candidates, cache = exp.candidates_with_clusters(candidate_ids, baseline)
    maximum = len(candidates)
    enriched = [{**item, "pair_vector": ((cache[item["img1"]].numpy() + cache[item["img2"]].numpy()) / 2).tolist()} for item in candidates]
    labelled = [{"img1": exp.groups[pair_id].iloc[0].resolved_img1, "img2": exp.groups[pair_id].iloc[0].resolved_img2} for pair_id in exp.initial_ids[:exp.cfg.initial_pairs]]
    labelled_cache = build_embedding_cache(labelled, baseline, exp.device, exp.cfg.symmetry_mode)
    labelled_vectors = np.stack([((labelled_cache[x["img1"]].numpy() + labelled_cache[x["img2"]].numpy()) / 2) for x in labelled])
    entropy = uncertainty_sampling(candidates, baseline, maximum, exp.device, exp.cfg.seed, cache)
    mc_scores, _ = score_mc_dropout(candidates, baseline, exp.device, cache, seed=exp.cfg.seed)
    return {
        "random": [row["pair_id"] for row in random_sampling(candidates, baseline, maximum, exp.device, exp.cfg.seed, cache)],
        "predictive_entropy": [row["pair_id"] for row in entropy],
        "core_set_k_center": [row["pair_id"] for row in core_set_select(enriched, labelled_vectors, maximum)],
        "mc_dropout_mutual_information": [row["pair_id"] for row in select_mc_dropout(mc_scores, "mc_dropout_mutual_information", maximum)],
        "mc_dropout_probability_variance": [row["pair_id"] for row in select_mc_dropout(mc_scores, "mc_dropout_probability_variance", maximum)],
    }


def run_capacity_audit(protocol: dict[str, Any], output: Path, device: str) -> None:
    rows = []
    for seed in protocol["seeds"]:
        try:
            exp = make_experiment(protocol, output, seed, "simclr", 1e-4, 3, 1e-4, device)
            initial, candidate = exp.load_and_split()
            rows.append({"seed": seed, "status": "pass", "initial_pair_groups": len(initial), "candidate_pair_groups": len(candidate)})
        except RuntimeError as error:
            rows.append({"seed": seed, "status": "fail", "reason": str(error)})
    pd.DataFrame(rows).to_csv(output / "audit" / "four_way_capacity_summary.csv", index=False)
    if any(row["status"] != "pass" for row in rows):
        raise RuntimeError("Capacity audit failed; no GPU training may be started. See four_way_capacity_summary.csv.")


def task_file(output: Path, kind: str, values: list[dict[str, Any]]) -> Path:
    path = output / "task_manifests" / f"{kind}.json"
    atomic_json({"kind": kind, "tasks": values}, path)
    return path


def run_validation_cell(protocol: dict[str, Any], output: Path, spec: dict[str, Any], device: str) -> None:
    identifier = stable_id(spec); path = output / "validation_cells" / f"{identifier}.json"
    if path.exists():
        print(f"completed validation cell: {path}", flush=True); return
    exp = make_experiment(protocol, output, **spec, device=device)
    initial, _ = exp.load_and_split()
    checkpoint = output / "resumable_checkpoints" / "validation" / f"{identifier}.pth"
    model, metrics, paused = train_with_epoch_checkpoints(exp, initial, checkpoint, f"validation-{identifier}", None, heartbeat_seconds=1800, checkpoint_enabled=True)
    if paused or model is None: raise RuntimeError("Validation cell paused; rerun the identical command to resume.")
    atomic_json({**spec, "validation_ideal_accuracy": exp.validation_accuracy(model), "validation_pairwise_accuracy": exp.pairwise_accuracy(model, exp.validation_rows), "training_pairwise_accuracy": metrics["pairwise_accuracy"], "checkpoint": str(checkpoint)}, path)


def write_colab_task_manifest(protocol: dict[str, Any], output: Path) -> None:
    validation = [{"seed": seed, "encoder": encoder, "learning_rate": lr, "epochs": epoch, "weight_decay": protocol["default_weight_decay"]} for seed in protocol["seeds"] for encoder in protocol["encoder_initializations"] for lr in protocol["learning_rates"] for epoch in protocol["epochs"]]
    task_file(output, "validation_learning_rate_epoch_cells", validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit_four_way_image_disjoint_capacity", "write_validation_learning_rate_epoch_task_manifest", "run_validation_cell"))
    parser.add_argument("--config", type=Path, default=HERE / "literature_backed_strict_image_disjoint_budget_curve_settings.json")
    parser.add_argument("--drive-output", type=Path, required=True, help="Mounted Google Drive output root; never use transient /content.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int); parser.add_argument("--encoder", choices=("simclr", "imagenet")); parser.add_argument("--learning-rate", type=float); parser.add_argument("--epochs", type=int); parser.add_argument("--weight-decay", type=float)
    args = parser.parse_args(); protocol = read_config(args.config); output = args.drive_output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.action == "audit_four_way_image_disjoint_capacity": run_capacity_audit(protocol, output, args.device)
    elif args.action == "write_validation_learning_rate_epoch_task_manifest": write_colab_task_manifest(protocol, output)
    else:
        needed = (args.seed, args.encoder, args.learning_rate, args.epochs, args.weight_decay)
        if any(value is None for value in needed): parser.error("run_validation_cell requires --seed --encoder --learning-rate --epochs --weight-decay")
        run_validation_cell(protocol, output, {"seed": args.seed, "encoder": args.encoder, "learning_rate": args.learning_rate, "epochs": args.epochs, "weight_decay": args.weight_decay}, args.device)


if __name__ == "__main__":
    main()

"""Data loading and image-disjoint partitioning for RHEED labels."""
from __future__ import annotations

import json
from pathlib import Path
import random
import pandas as pd

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_RECONSTRUCTION_TYPES = {"(1 x 1)", "Twinned(2 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR"}


def original_data_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    return root / "original data" if (root / "original data").exists() else root


def find_source_csv(root: Path, stem: str) -> Path:
    choices = sorted(root.glob(f"{stem}*.csv"))
    if not choices:
        raise FileNotFoundError(f"No {stem} CSV found below {root}")
    return choices[-1]


def resolve_image_path(root: Path, value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = root / value
    if candidate.exists():
        return str(candidate.resolve())
    matches = list(root.rglob(Path(value).name))
    return str(matches[0].resolve()) if len(matches) == 1 else None


def load_pairwise_rows(data_root: str | Path) -> list[dict]:
    root = original_data_root(data_root)
    frame = pd.read_csv(find_source_csv(root, "Quantum Label Data - Pairwise_Comparison"))
    rows: list[dict] = []
    for index, row in frame.iterrows():
        left = resolve_image_path(root, row.get("Image1_Path", row.get("Image1_Name")))
        right = resolve_image_path(root, row.get("Image2_Path", row.get("Image2_Name")))
        label = str(row.get("Winner", "")).strip().lower()
        reconstruction = str(row.get("Reconstruction_Type", row.get("reconstruction_type", ""))).strip()
        if left and right and label in {"1", "2", "tie", "not_apply"} and reconstruction in SUPPORTED_RECONSTRUCTION_TYPES:
            rows.append({"pair_id": f"pair_{index:04d}", "left": left, "right": right,
                         "winner": label, "reconstruction_type": reconstruction})
    if not rows:
        raise ValueError("No usable pairwise rows were found.")
    return rows


def load_absolute_and_ideal_anchors(data_root: str | Path) -> list[dict]:
    """Load direct bad-image labels and ideal-reference reconstruction anchors."""
    root = original_data_root(data_root); anchors: list[dict] = []
    try:
        frame = pd.read_csv(find_source_csv(root, "Quantum Label Data - Absolute_Scoring"))
        for _, row in frame.iterrows():
            path = resolve_image_path(root, row.get("File_Path", row.get("File_Name")))
            reconstruction = str(row.get("Reconstruction", ""))
            if path and "Bad" in reconstruction:
                anchors.append({"path": path, "quality_target": 0., "reconstruction_type": None, "source": "absolute_bad"})
    except FileNotFoundError:
        pass
    directory_labels = {"STO_ideal_1x1": "(1 x 1)", "STO_ideal_Twinned2x1": "Twinned(2 x 1)",
                        "STO_ideal_c6x2": "c(6 x 2)", "STO_ideal_RT13": "(√13 x √13)", "STO_ideal_HTR": "HTR"}
    for directory, label in directory_labels.items():
        for path in (root / directory).glob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                anchors.append({"path": str(path.resolve()), "quality_target": 1., "reconstruction_type": label, "source": "ideal_reference"})
    return anchors


def create_image_disjoint_split(pair_rows: list[dict], output_path: str | Path, seed: int = 42,
                                test_fraction: float = .20, validation_fraction: float = .20) -> dict:
    """Allocate image identities before any model-related work; pairs crossing groups are excluded."""
    output_path = Path(output_path)
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    images = sorted({row[side] for row in pair_rows for side in ("left", "right")})
    random.Random(seed).shuffle(images)
    test_end = max(1, round(len(images) * test_fraction))
    validation_end = test_end + max(1, round(len(images) * validation_fraction))
    groups = {"test": sorted(images[:test_end]), "validation": sorted(images[test_end:validation_end]), "train": sorted(images[validation_end:])}
    membership = {image: group for group, members in groups.items() for image in members}
    within_group_pairs = {group: [r["pair_id"] for r in pair_rows if membership[r["left"]] == group and membership[r["right"]] == group] for group in groups}
    value = {"seed": seed, "test_image_policy": "Test image identities are excluded from all training, validation, augmentation and feature fitting.",
             "images": groups, "pair_ids_within_group": within_group_pairs}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


def session_id_for_image(path: str | Path) -> str:
    """Return the acquisition-session directory below ``Trajectories``.

    A dated acquisition session is deliberately the unit held out in the
    definitive study.  Adjacent frames in one growth run are visually similar,
    so holding out only image filenames can make a model look better than it
    will be on a later experiment.
    """
    image_path = Path(path)
    parts = image_path.parts
    try:
        return parts[parts.index("Trajectories") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Cannot infer an acquisition session from {image_path}") from error


def session_held_out_audit(pair_rows: list[dict], minimum_decisive_pairs: int = 50) -> dict:
    """Count only within-session pairs that can be used as a sealed test fold."""
    sessions = sorted({session_id_for_image(row["left"]) for row in pair_rows} |
                      {session_id_for_image(row["right"]) for row in pair_rows})
    folds = []
    for session in sessions:
        rows = [row for row in pair_rows
                if session_id_for_image(row["left"]) == session
                and session_id_for_image(row["right"]) == session]
        decisive = [row for row in rows if row["winner"] in {"1", "2"}]
        by_label = {label: sum(row["winner"] == label for row in rows)
                    for label in ("1", "2", "tie", "not_apply")}
        folds.append({"held_out_session": session, "all_pair_count": len(rows),
                      "decisive_pair_count": len(decisive), "labels": by_label,
                      "minimum_decisive_pairs": minimum_decisive_pairs,
                      "additional_decisive_pairs_needed": max(0, minimum_decisive_pairs - len(decisive)),
                      "eligible": len(decisive) >= minimum_decisive_pairs})
    return {"protocol": "session_held_out", "minimum_decisive_pairs_per_outer_test_fold": minimum_decisive_pairs,
            "folds": folds, "eligible_to_train": bool(folds) and all(fold["eligible"] for fold in folds),
            "interpretation": "A fold is eligible only when its untouched acquisition session has enough decisive expert comparisons to estimate accuracy."}


def create_session_held_out_split(pair_rows: list[dict], held_out_session: str,
                                  output_path: str | Path, seed: int = 42,
                                  validation_fraction: float = .20) -> dict:
    """Seal one acquisition session and create image-disjoint validation data from the others."""
    output_path = Path(output_path)
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    test_images = sorted({image for row in pair_rows for image in (row["left"], row["right"])
                          if session_id_for_image(image) == held_out_session})
    candidate_images = sorted({image for row in pair_rows for image in (row["left"], row["right"])
                               if session_id_for_image(image) != held_out_session})
    random.Random(seed).shuffle(candidate_images)
    validation_size = max(1, round(len(candidate_images) * validation_fraction))
    groups = {"test": test_images, "validation": sorted(candidate_images[:validation_size]),
              "train": sorted(candidate_images[validation_size:])}
    membership = {image: group for group, images in groups.items() for image in images}
    within_group_pairs = {
        group: [row["pair_id"] for row in pair_rows
                if membership.get(row["left"]) == group and membership.get(row["right"]) == group]
        for group in groups
    }
    value = {
        "seed": seed, "protocol": "session_held_out_outer_fold",
        "held_out_test_session": held_out_session,
        "test_image_policy": "All images from the held-out acquisition session are excluded from training, validation, augmentation, calibration, feature fitting, and architecture selection.",
        "images": groups, "pair_ids_within_group": within_group_pairs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


def create_fixed_five_fold_unseen_image_splits(pair_rows: list[dict], output_path: str | Path,
                                               seed: int = 42) -> dict:
    """Create five deterministic folds with an unseen image endpoint in every test pair.

    This is an image-generalisation protocol, not a session-generalisation
    protocol.  A test pair is retained whenever at least one endpoint is a
    held-out image.  That endpoint is never used by pairwise fine-tuning,
    validation calibration, or architecture selection.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    images = sorted({image for row in pair_rows for image in (row["left"], row["right"])})
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    groups = [set(shuffled[index::5]) for index in range(5)]
    folds = []
    for fold_index in range(5):
        test_images = groups[fold_index]
        validation_images = groups[(fold_index + 1) % 5]
        train_images = set(images) - test_images - validation_images
        roles = {
            "train": [row["pair_id"] for row in pair_rows
                      if row["left"] in train_images and row["right"] in train_images],
            "validation": [row["pair_id"] for row in pair_rows
                           if row["left"] not in test_images and row["right"] not in test_images
                           and (row["left"] in validation_images or row["right"] in validation_images)],
            "test": [row["pair_id"] for row in pair_rows
                     if row["left"] in test_images or row["right"] in test_images],
        }
        folds.append({
            "fold_index": fold_index,
            "test_image_policy": "Every test pair contains at least one image absent from pairwise fine-tuning, validation, calibration, and architecture selection.",
            "images": {"train": sorted(train_images), "validation": sorted(validation_images), "test": sorted(test_images)},
            "pair_ids_by_role": roles,
            "pair_counts": {role: len(pair_ids) for role, pair_ids in roles.items()},
            "decisive_pair_counts": {role: sum(row["pair_id"] in set(pair_ids) and row["winner"] in {"1", "2"} for row in pair_rows)
                                      for role, pair_ids in roles.items()},
        })
    value = {
        "protocol": "fixed_five_fold_unseen_image_endpoint",
        "split_seed": seed,
        "pretraining_policy": "The frozen laboratory RHEED-SimCLR encoder may have self-supervised exposure to all RHEED images, but never uses pairwise preference labels. Image exclusion applies to pairwise reward-model fine-tuning and model selection.",
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value

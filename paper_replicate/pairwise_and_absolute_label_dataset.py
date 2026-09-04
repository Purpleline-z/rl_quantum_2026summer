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

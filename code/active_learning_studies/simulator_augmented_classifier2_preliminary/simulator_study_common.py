"""Shared implementation for the simulator-augmented Classifier2 preliminary study.

This code intentionally lives outside ``classifier2/``: that directory contains
the PhD-authored reference documentation and is not modified by this study.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
PROTOCOL = json.loads((HERE / "fixed_protocol.json").read_text(encoding="utf-8"))
ALL_TYPES = ["(1 x 1)", "Twinned(2 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR"]
TYPE_INDEX = {name: index for index, name in enumerate(ALL_TYPES)}
SYNTHETIC_TO_REAL = {"twinned_2x1": "Twinned(2 x 1)", "c_6x2": "c(6 x 2)", "rt13": "(√13 x √13)"}
IDEAL_DIRECTORIES = {
    "(1 x 1)": "STO_ideal_1x1", "Twinned(2 x 1)": "STO_ideal_Twinned2x1",
    "c(6 x 2)": "STO_ideal_c6x2", "(√13 x √13)": "STO_ideal_RT13", "HTR": "STO_ideal_HTR",
}


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def resolve_data_root(value: str | Path | None = None) -> Path:
    root = Path(value or DEFAULT_DATA_ROOT).expanduser().resolve()
    if not (root / "original data").is_dir():
        raise FileNotFoundError(f"Expected 'original data' under data root: {root}")
    return root


def archive_member_root(archive: Path) -> str:
    with zipfile.ZipFile(archive) as bundle:
        candidates = [name for name in bundle.namelist() if name.endswith("manifest.json")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one manifest.json in synthetic archive, found {candidates}")
    return candidates[0].rsplit("/", 1)[0]


def validate_synthetic_archive(archive: Path, verify_image_hashes: bool = True) -> dict:
    """Validate the supplied asset without extracting it or allocating image arrays."""
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    progress(f"Validating synthetic archive: {archive.name}")
    with zipfile.ZipFile(archive) as bundle:
        root = archive_member_root(archive)
        manifest = json.loads(bundle.read(f"{root}/manifest.json"))
        metadata_rows = list(csv.DictReader(bundle.read(f"{root}/metadata.csv").decode("utf-8").splitlines()))
        required_manifest = {"generator", "class_counts", "image_count", "generator_config", "detector_contract"}
        missing = required_manifest - set(manifest)
        if missing:
            raise ValueError(f"Synthetic manifest missing fields: {sorted(missing)}")
        if manifest["generator"] != "classifier_synthetic_sto_v6" or manifest["generator_config"].get("schema_version") != 6:
            raise ValueError("Only schema-v6 classifier synthetic data are accepted.")
        if manifest["detector_contract"]["output"] != {"bit_depth": 8, "dtype": "uint8", "height": 492, "width": 656}:
            raise ValueError("Preliminary protocol accepts only native 656x492 uint8 data.")
        labels = {row["reconstruction_label"] for row in metadata_rows}
        if labels != set(SYNTHETIC_TO_REAL):
            raise ValueError(f"Unexpected synthetic labels: {sorted(labels)}")
        if not all(str(row.get("synthetic_training_only", "")).lower() == "true" for row in metadata_rows):
            raise ValueError("Every synthetic row must explicitly be training-only.")
        if len(metadata_rows) != int(manifest["image_count"]):
            raise ValueError("Metadata row count does not match manifest image_count.")
        groups = pd.DataFrame(metadata_rows).groupby("same_surface_group")
        group_sizes = groups.size()
        if not (group_sizes == 3).all() or groups["reconstruction_label"].nunique().max() != 1:
            raise ValueError("Each same_surface_group must contain exactly three views from one label.")
        names = set(bundle.namelist())
        missing_images = [row["image_path"] for row in metadata_rows if f"{root}/{row['image_path']}" not in names]
        if missing_images:
            raise ValueError(f"Archive metadata references missing images, e.g. {missing_images[:3]}")
        if verify_image_hashes:
            for index, row in enumerate(metadata_rows, start=1):
                payload = bundle.read(f"{root}/{row['image_path']}")
                if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
                    raise ValueError(f"Image hash mismatch: {row['image_path']}")
                if index % 250 == 0 or index == len(metadata_rows):
                    progress(f"Archive image-hash progress {index}/{len(metadata_rows)}")
    return {
        "status": "pass", "archive": str(archive), "archive_sha256": sha256_file(archive),
        "archive_member_root": root, "generator": manifest["generator"], "schema_version": 6,
        "synthetic_image_count": len(metadata_rows), "class_counts": manifest["class_counts"],
        "same_surface_group_count": int(len(group_sizes)), "images_per_group": int(group_sizes.iloc[0]),
        "raster": manifest["detector_contract"]["output"], "synthetic_training_only": True,
        "image_hashes_verified": verify_image_hashes,
    }


def extract_synthetic_archive(archive: Path, cache_root: Path) -> Path:
    """Extract once into ephemeral storage; reject ZIP path traversal."""
    root = archive_member_root(archive)
    destination = cache_root / root
    marker = destination / "manifest.json"
    if marker.is_file():
        progress(f"Using existing ephemeral synthetic extraction: {destination}")
        return destination
    cache_root.mkdir(parents=True, exist_ok=True)
    progress(f"Extracting synthetic archive into ephemeral storage: {cache_root}")
    with zipfile.ZipFile(archive) as bundle:
        for index, info in enumerate(bundle.infolist(), start=1):
            target = (cache_root / info.filename).resolve()
            if cache_root.resolve() not in target.parents and target != cache_root.resolve():
                raise ValueError(f"Unsafe archive member: {info.filename}")
            bundle.extract(info, cache_root)
            if index % 500 == 0 or index == len(bundle.infolist()):
                progress(f"Extraction progress {index}/{len(bundle.infolist())}")
    if not marker.is_file():
        raise RuntimeError("Synthetic extraction completed without manifest.json")
    return destination


def list_ideal_images(data_root: Path) -> dict[str, list[Path]]:
    source = data_root / "original data"
    result: dict[str, list[Path]] = {}
    for name, directory in IDEAL_DIRECTORIES.items():
        result[name] = sorted(path.resolve() for path in (source / directory).glob("*") if path.is_file())
    return result


def build_real_partitions(data_root: Path, output: Path) -> dict:
    """Build a fixed strict image-disjoint partition based on labelled ideal images."""
    random_generator = random.Random(int(PROTOCOL["real_partition_seed"]))
    ideals = list_ideal_images(data_root)
    records = []
    for label, paths in ideals.items():
        hash_groups: dict[str, list[Path]] = {}
        for path in paths:
            hash_groups.setdefault(sha256_file(path), []).append(path)
        shuffled = list(hash_groups.items()); random_generator.shuffle(shuffled)
        if label in PROTOCOL["real_evaluation_types"]:
            if len(shuffled) < 4:
                raise ValueError(f"Need at least four unique ideal-image hashes for train/validation/test: {label}")
            validation_count = max(1, round(len(shuffled) * 0.2))
            test_count = max(1, round(len(shuffled) * 0.2))
            train_count = len(shuffled) - validation_count - test_count
            if train_count < 2:
                raise ValueError(f"Insufficient training anchors for {label}")
            partitions = (["outer_test"] * test_count + ["validation"] * validation_count + ["train"] * train_count)
        else:
            partitions = ["train"] * len(shuffled)
        for (image_hash, paths_for_hash), partition in zip(shuffled, partitions):
            for path in paths_for_hash:
                records.append({"image_id": str(path), "image_sha256": image_hash, "label": label, "partition": partition, "source": "ideal_reference"})
    table = pd.DataFrame(records)
    if table.image_id.duplicated().any() or (table.groupby("image_sha256").partition.nunique() > 1).any():
        raise ValueError("An image ID or duplicate-image hash crosses partitions.")
    overlaps = {left + "_" + right: int(len(set(table.loc[table.partition == left, "image_sha256"]) & set(table.loc[table.partition == right, "image_sha256"]))) for left in ("train", "validation", "outer_test") for right in ("train", "validation", "outer_test") if left < right}
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "strict_real_image_partitions.csv", index=False)
    counts = {f"{partition}|{label}": int(count) for (partition, label), count in table.groupby(["partition", "label"]).size().items()}
    audit = {"status": "pass", "partition_seed": PROTOCOL["real_partition_seed"], "counts_by_partition_and_label": counts, "image_id_overlap_counts": overlaps}
    atomic_json_write(audit, output / "strict_real_image_partition_audit.json")
    return audit


def load_partition_table(drive_output: Path) -> pd.DataFrame:
    table = drive_output / "preflight_audits" / "strict_real_image_partitions.csv"
    if not table.is_file():
        raise FileNotFoundError(f"Run build_strict_real_image_partitions.py first: {table}")
    return pd.read_csv(table)


def resolve_original_file(data_root: Path, relative: str) -> Path:
    path = (data_root / "original data" / str(relative).replace("\\", "/")).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def canonical_type(value: str) -> str | None:
    text = str(value).strip()
    if text in TYPE_INDEX: return text
    if "13" in text or "RT13" in text.upper(): return "(√13 x √13)"
    return None


def load_real_training_inputs(data_root: Path, partition_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[Path]], list[Path], list[Path]]:
    excluded = set(partition_table.loc[partition_table.partition != "train", "image_id"])
    source = data_root / "original data"
    pairwise = pd.read_csv(source / "Quantum Label Data - Pairwise_Comparisonv1.8.csv")
    pairwise["resolved_img1"] = pairwise.Image1_Path.map(lambda value: str(resolve_original_file(data_root, value)))
    pairwise["resolved_img2"] = pairwise.Image2_Path.map(lambda value: str(resolve_original_file(data_root, value)))
    pairwise["canonical_type"] = pairwise.Reconstruction_Type.map(canonical_type)
    pairwise = pairwise[pairwise.canonical_type.notna()].copy()
    pairwise = pairwise[~pairwise.resolved_img1.isin(excluded) & ~pairwise.resolved_img2.isin(excluded)].copy()
    anchors = {label: [Path(path) for path in partition_table[(partition_table.partition == "train") & (partition_table.label == label)].image_id] for label in ALL_TYPES}
    absolute = pd.read_csv(source / "Quantum Label Data - Absolute_Scoringv1.8 (1).csv")
    bad = []
    for _, row in absolute[absolute.Reconstruction.astype(str).str.contains("Bad", na=False)].iterrows():
        path = resolve_original_file(data_root, row.File_Path)
        if str(path) not in excluded: bad.append(path)
    trajectories = sorted({Path(path) for path in pairwise.resolved_img1} | {Path(path) for path in pairwise.resolved_img2})
    if not len(pairwise) or not trajectories:
        raise ValueError("Strict split leaves no usable pairwise real training rows.")
    return pairwise.reset_index(drop=True), anchors, bad, trajectories


class GrayTransform:
    def __init__(self):
        self.value = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), T.Normalize([.5] * 3, [.25] * 3)])
    def __call__(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.value(image.convert("L"))


class FiveOutputClassifier2CompatibleModel(nn.Module):
    """ResNet-18 and 512→256→5 reward head compatible with Classifier2."""
    def __init__(self, simclr_checkpoint: Path):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        state = torch.load(simclr_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        cleaned = {str(key).replace("encoder.", ""): value for key, value in state.items() if not str(key).startswith("projector.")}
        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        progress(f"Loaded shipped SimCLR encoder ({len(missing)} missing, {len(unexpected)} unexpected keys).")
        self.reward_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, 5))
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.reward_head(self.encoder(images).flatten(1))


class SyntheticDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, root: Path): self.rows, self.root, self.tf = rows.reset_index(drop=True), root, GrayTransform()
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return self.tf(self.root / row.image_path), TYPE_INDEX[SYNTHETIC_TO_REAL[row.reconstruction_label]]


class RealUnifiedDataset(Dataset):
    def __init__(self, pairwise: pd.DataFrame, anchors: dict[str, list[Path]], bad: list[Path], trajectories: list[Path], samples_per_epoch: int, seed: int):
        self.pairwise, self.anchors, self.bad, self.trajectories = pairwise, anchors, bad, trajectories
        self.samples_per_epoch, self.seed, self.tf = samples_per_epoch, seed, GrayTransform()
    def __len__(self): return self.samples_per_epoch
    def load(self, path: Path): return self.tf(path)
    def __getitem__(self, index):
        rng = random.Random(self.seed * 1000003 + index + random.randint(0, 2**31 - 1))
        roll = rng.random()
        if roll < .60:
            row = self.pairwise.iloc[rng.randrange(len(self.pairwise))]; a, b = Path(row.resolved_img1), Path(row.resolved_img2); winner = str(row.Winner)
            target = {"1": 0, "2": 1, "tie": 2, "not_apply": 3}.get(winner, 3)
            return "pairwise", self.load(a), self.load(b), TYPE_INDEX[row.canonical_type], target, 1.0
        if roll < .85 or not self.bad:
            available = [key for key, values in self.anchors.items() if values]
            label = available[rng.randrange(len(available))]; ideal = self.anchors[label][rng.randrange(len(self.anchors[label]))]; other = self.trajectories[rng.randrange(len(self.trajectories))]
            return "pairwise", self.load(ideal), self.load(other), TYPE_INDEX[label], 0, 1.0
        bad = self.bad[rng.randrange(len(self.bad))]; other = self.trajectories[rng.randrange(len(self.trajectories))]
        return "pairwise", self.load(other), self.load(bad), rng.randrange(5), 0, 1.0


def real_collate(batch):
    _, a, b, type_index, winner, weight = zip(*batch)
    return torch.stack(a), torch.stack(b), torch.tensor(type_index), torch.tensor(winner), torch.tensor(weight, dtype=torch.float32)


def pairwise_loss(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    first, second, type_index, winner, weight = [value.to(device) for value in batch]
    r1, r2 = model(first), model(second); indices = torch.arange(len(first), device=device)
    a, b = r1[indices, type_index], r2[indices, type_index]
    losses = torch.where(winner == 0, -F.logsigmoid(a-b), torch.where(winner == 1, -F.logsigmoid(b-a), torch.where(winner == 2, torch.abs(a-b), F.relu(a)+F.relu(b))))
    return (losses * weight).mean()


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, best_metric: float, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metric": best_metric, "stage": stage}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer) -> tuple[int, float]:
    if not path.is_file(): return 0, -float("inf")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
    progress(f"Resuming {state['stage']} from epoch {state['epoch']}")
    return int(state["epoch"]), float(state["best_metric"])


def evaluate_three_class(model: nn.Module, partition: pd.DataFrame, device: torch.device, partition_name: str = "outer_test") -> dict:
    test = partition[partition.partition == partition_name]
    tf = GrayTransform(); labels = list(PROTOCOL["real_evaluation_types"]); indices = [TYPE_INDEX[label] for label in labels]
    truth, prediction = [], []
    model.eval()
    with torch.no_grad():
        for _, row in test.iterrows():
            scores = model(tf(Path(row.image_id)).unsqueeze(0).to(device))[0, indices].cpu().numpy()
            truth.append(labels.index(row.label)); prediction.append(int(np.argmax(scores)))
    per_class = {}
    for index, label in enumerate(labels):
        count = sum(value == index for value in truth); correct = sum(a == b == index for a, b in zip(truth, prediction))
        per_class[label] = {"correct": correct, "total": count, "accuracy": None if count == 0 else correct / count}
    accuracy = float(np.mean(np.asarray(truth) == np.asarray(prediction))) if truth else float("nan")
    recalls = [value["accuracy"] for value in per_class.values() if value["accuracy"] is not None]
    return {f"{partition_name}_accuracy": accuracy, f"{partition_name}_macro_recall": float(np.mean(recalls)), "per_class": per_class, "confusion_matrix": [[sum(t == i and p == j for t, p in zip(truth, prediction)) for j in range(3)] for i in range(3)], "labels": labels}


def synthetic_rows(synthetic_root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(synthetic_root / "metadata.csv")
    groups = rows[["same_surface_group", "reconstruction_label"]].drop_duplicates().sort_values("same_surface_group").reset_index(drop=True)
    rng = random.Random(seed); train_groups, validation_groups = set(), set()
    for label, group in groups.groupby("reconstruction_label"):
        values = group.same_surface_group.tolist(); rng.shuffle(values); cut = round(len(values) * (1-PROTOCOL["synthetic_pretraining"]["synthetic_validation_fraction"])); train_groups.update(values[:cut]); validation_groups.update(values[cut:])
    train, validation = rows[rows.same_surface_group.isin(train_groups)].copy(), rows[rows.same_surface_group.isin(validation_groups)].copy()
    if set(train.same_surface_group) & set(validation.same_surface_group): raise AssertionError("Synthetic group leakage")
    return train, validation

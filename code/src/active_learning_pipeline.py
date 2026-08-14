#!/usr/bin/env python3
"""Reproducible active learning for RHEED pairwise comparisons.

The CSV preference winner is deliberately distinct from the acquisition utility.
When a pair is acquired, its CSV rows train the Bradley--Terry (BT) model.  Its
utility label is the change in fixed ideal-image accuracy after controlled
retraining with that pair.  This makes the simulation answer the downstream
classification question rather than self-confirming BT predictions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

HERE = Path(__file__).resolve().parent
from project_paths import CODE_ROOT, RESULT_ROOT, resolve_data_root
from dataset_protocol import dataset_spec, file_hashes, locate_input
ROOT = CODE_ROOT
sys.path.insert(0, str(HERE))
from strategies import (build_embedding_cache, random_sampling, uncertainty_sampling,
    cluster_quota_uncertainty_sampling, cluster_margin_pairwise_sampling, score_uncertainty)
from mc_dropout_uncertainty import score_mc_dropout, select_mc_dropout
from selection_utils import core_set_select, pair_vector as cached_pair_vector, uncertainty_diversity_select
from advanced_features import (SYMMETRY_MODES, fit_metadata_model, load_metadata,
    metadata_vector_for_path, mixture_metrics, normalized, symmetry_image)

TYPE_ORDER = ["(1 x 1)", "Twinned(2 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR"]
TYPE_TO_INDEX = {name: i for i, name in enumerate(TYPE_ORDER)}
RAW_TYPE_MAP = {
    "(1 x 1)": "(1 x 1)", "Twinned(2 x 1)": "Twinned(2 x 1)",
    "c(6 x 2)": "c(6 x 2)", "HTR": "HTR",
}
IDEAL_DIRS = {
    "(1 x 1)": "STO_ideal_1x1", "Twinned(2 x 1)": "STO_ideal_Twinned2x1",
    "c(6 x 2)": "STO_ideal_c6x2", "(√13 x √13)": "STO_ideal_RT13", "HTR": "STO_ideal_HTR",
}


def canonical_type(value: object) -> str | None:
    """Normalize mojibake variants of the root-13 label without touching CSV data."""
    text = str(value).strip()
    if text in RAW_TYPE_MAP:
        return RAW_TYPE_MAP[text]
    compact = text.lower().replace(" ", "").replace("√", "root").replace("鈭?", "root")
    if "13" in compact or "rt13" in compact or "root13" in compact:
        return "(√13 x √13)"
    return None


def canonical_pair(a: str, b: str) -> str:
    return " || ".join(sorted((str(a).replace("\\", "/"), str(b).replace("\\", "/"))))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform(symmetry_mode: str = "none") -> T.Compose:
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize([.5] * 3, [.25] * 3)])


class PairRows(Dataset):
    def __init__(self, rows: pd.DataFrame, symmetry_mode: str = "none", metadata_for_path=None):
        self.rows, self.tf, self.symmetry_mode = rows.reset_index(drop=True), transform(symmetry_mode), symmetry_mode
        self.metadata_for_path = metadata_for_path
    def __len__(self): return len(self.rows)
    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        with Image.open(r.resolved_img1) as a, Image.open(r.resolved_img2) as b:
            values = (self.tf(symmetry_image(a, self.symmetry_mode)), self.tf(symmetry_image(b, self.symmetry_mode)), int(r.type_idx),
                      str(r.Winner), float(r.confidence_weight))
            if self.metadata_for_path is None:
                return values
            return (*values, torch.tensor(self.metadata_for_path(r.resolved_img1), dtype=torch.float32),
                    torch.tensor(self.metadata_for_path(r.resolved_img2), dtype=torch.float32))


class FlattenedEncoder(nn.Sequential):
    """ResNet feature stack whose public interface is [batch, 512]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).flatten(1)


class BTModel(nn.Module):
    """ResNet-18 reward model compatible with the supplied SimCLR encoder."""
    def __init__(self, encoder_weights: Path | None, hidden_dim: int = 256, dropout_p: float = .2,
                 metadata_dim: int = 0, encoder_initialization: str = "simclr"):
        super().__init__()
        if encoder_initialization == "imagenet":
            # Download/caching is delegated to torchvision.  The immutable run
            # manifest records this named public initialization separately from
            # the local SimCLR checkpoint hash.
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = FlattenedEncoder(*list(backbone.children())[:-1])
        elif encoder_initialization == "simclr":
            self.encoder = FlattenedEncoder(*list(models.resnet18(weights=None).children())[:-1])
        else:
            raise ValueError("encoder_initialization must be 'simclr' or 'imagenet'.")
        if encoder_initialization == "simclr" and encoder_weights and encoder_weights.exists():
            state = torch.load(encoder_weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
            cleaned = {k.replace("encoder.", ""): v for k, v in state.items() if not k.startswith("projector.")}
            missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
            print(f"Loaded SimCLR encoder ({len(missing)} missing, {len(unexpected)} unexpected keys).", flush=True)
        self.metadata_dim = metadata_dim
        if metadata_dim:
            self.metadata_branch = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
            self.fusion = nn.Sequential(nn.Linear(512 + 64, 512), nn.ReLU(inplace=True))
        else:
            self.metadata_branch = self.fusion = None
        self.reward_head = nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_p), nn.Linear(hidden_dim, len(TYPE_ORDER)))

    def reward_from_embeddings(self, embeddings: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        if self.metadata_dim:
            if metadata is None or metadata.shape[-1] != self.metadata_dim:
                raise ValueError(f"Metadata fusion requires vectors with width {self.metadata_dim}.")
            embeddings = self.fusion(torch.cat((embeddings, self.metadata_branch(metadata)), dim=-1))
        return self.reward_head(embeddings)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        return self.reward_from_embeddings(self.encoder(x), metadata)


@dataclass
class Config:
    #initial_pairs: int = 70; candidate_pairs: int = 100; budget: int = 50; batch_size: int = 5
    initial_pairs: int = 50; candidate_pairs: int = 120; budget: int = 50; batch_size: int = 5
    epochs: int = 10; train_batch_size: int = 16; lr: float = 1e-4; weight_decay: float = 1e-4
    test_fraction: float = .2; clusters: int = 20; seed: int = 42; device: str = "auto"
    strategies: str = "random,uncertainty,cluster_quota_uncertainty"; include_twinned: bool = False
    utility_per_pair: bool = True; utility_min_history: int = 10; force: bool = False
    dropout_p: float = .2; mc_samples: int = 20
    acquisition_mode: str = "sequential"
    diversity_lambda: float = .5
    symmetry_mode: str = "none"
    metadata_csv: str | None = None
    metadata_weight: float = .5
    mixture_weight: float = .5
    manifest_dir: str | None = None
    data_root: str | None = None
    dataset_version: str = "v1.8"
    utility_validation_fraction: float = .2
    bad_anchor_weight: float = .10
    encoder_initialization: str = "simclr"


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.dataset = dataset_spec(cfg.dataset_version)
        self.data_root = resolve_data_root(cfg.data_root) / "original data"
        # Resolve only when a run loads data. This keeps construction usable for
        # path/configuration checks while still failing closed at experiment time.
        self.pairwise_csv = self.data_root / self.dataset.pairwise_name
        self.absolute_csv = self.data_root / self.dataset.absolute_name
        self.encoder_weights = ROOT / "classifier2" / "simclr_resnet18_encoder.pth"
        self.output = RESULT_ROOT / f"active_learning_{self.dataset.version}_seed{cfg.seed}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
        self.references: dict[str, list[Path]] = {}; self.test_images: dict[str, list[Path]] = {}; self.utility_images: dict[str, list[Path]] = {}
        self.groups: dict[str, pd.DataFrame] = {}; self.candidate_metadata: dict[str, dict[str, str]] = {}; self.bad_paths: list[Path] = []
        self.utility_cache_path = self.output / "utility_cache.json"
        self.utility_cache = json.loads(self.utility_cache_path.read_text()) if self.utility_cache_path.exists() else {}
        self.metadata: pd.DataFrame | None = None; self.metadata_columns: list[str] = []; self.metadata_model = None

    @property
    def metadata_dim(self) -> int:
        return 0 if self.metadata_model is None else 2 * len(self.metadata_columns)

    def metadata_vector(self, path: str) -> np.ndarray:
        if self.metadata is None or self.metadata_model is None:
            return np.empty((0,), dtype=np.float32)
        return metadata_vector_for_path(self.metadata, self.metadata_model, path).astype(np.float32)

    def load_and_split(self) -> tuple[list[str], list[str]]:
        self.pairwise_csv = locate_input(self.data_root.parent, ROOT, self.dataset.pairwise_name)
        self.absolute_csv = locate_input(self.data_root.parent, ROOT, self.dataset.absolute_name)
        df = pd.read_csv(self.pairwise_csv)
        required = {"Image1_Path", "Image2_Path", "Reconstruction_Type", "Winner"}
        if not required <= set(df): raise ValueError(f"CSV misses {required - set(df)}")
        df["canonical_type"] = df.Reconstruction_Type.map(canonical_type)
        df = df[df.canonical_type.notna() & df.Winner.astype(str).isin(["1", "2", "tie", "not_apply"])].copy()
        df["pair_id"] = [canonical_pair(a, b) for a, b in zip(df.Image1_Path, df.Image2_Path)]
        df["resolved_img1"] = [str((self.data_root / p).resolve()) for p in df.Image1_Path]
        df["resolved_img2"] = [str((self.data_root / p).resolve()) for p in df.Image2_Path]
        df = df[df.resolved_img1.map(lambda p: Path(p).is_file()) & df.resolved_img2.map(lambda p: Path(p).is_file())].copy()
        if df.empty: raise ValueError("No valid pairwise rows with resolvable images.")
        df["type_idx"] = df.canonical_type.map(TYPE_TO_INDEX)
        df["confidence_weight"] = df.get("Confidence", pd.Series(index=df.index)).map({"Confident": 1.0, "Somewhat sure": .7}).fillna(1.0)
        self.groups = {key: g.reset_index(drop=True) for key, g in df.groupby("pair_id", sort=True)}
        self._split_ideals()
        self._load_bad_references()
        ids = list(self.groups); rng = random.Random(self.cfg.seed); rng.shuffle(ids)
        # Greedy initial selection guarantees per-type coverage when possible.
        selected: list[str] = []; covered: set[str] = set()
        for pair_id in ids:
            types = set(self.groups[pair_id].canonical_type)
            if types - covered:
                selected.append(pair_id); covered |= types
        for pair_id in ids:
            if len(selected) >= min(self.cfg.initial_pairs, len(ids)): break
            if pair_id not in selected: selected.append(pair_id)
        remaining = [x for x in ids if x not in set(selected)]
        candidates = remaining[:min(self.cfg.candidate_pairs, len(remaining))]
        # Selection receives this label-free view only.  Human preference rows
        # remain in ``groups`` as the simulated, hidden acquisition oracle.
        self.candidate_metadata = {
            pair_id: {"pair_id": pair_id, "img1": str(self.groups[pair_id].iloc[0].resolved_img1),
                      "img2": str(self.groups[pair_id].iloc[0].resolved_img2)}
            for pair_id in candidates
        }
        self._write_manifests(selected, candidates)
        if self.cfg.metadata_csv:
            known = [p for g in self.groups.values() for p in g.resolved_img1.tolist() + g.resolved_img2.tolist()]
            self.metadata = load_metadata(self.cfg.metadata_csv, known)
            self.metadata_columns = [c for c in self.metadata.columns if c != "metadata_key" and pd.api.types.is_numeric_dtype(self.metadata[c])]
            train_paths = {p for pair_id in selected for p in self.groups[pair_id].resolved_img1.tolist() + self.groups[pair_id].resolved_img2.tolist()}
            train_meta = self.metadata[self.metadata.metadata_key.isin(train_paths)]
            if not self.metadata_columns or train_meta.empty: raise ValueError("Metadata has no matched numeric columns for the initial labelled split.")
            self.metadata_model = fit_metadata_model(train_meta, self.metadata_columns)
            pd.DataFrame([{"matched_rows": len(self.metadata), "training_rows": len(train_meta), "numeric_columns": ",".join(self.metadata_columns)}]).to_csv(self.output / "metadata_coverage.csv", index=False)
        overlap = self._image_overlap(selected, candidates)
        print(f"Validated {len(df)} rows / {len(self.groups)} pairs. Pretrain={len(selected)}, candidate={len(candidates)}, pairwise image overlap={len(overlap)}.", flush=True)
        if set(TYPE_ORDER) - covered: print(f"WARNING: missing initial coverage for {set(TYPE_ORDER) - covered}", flush=True)
        return selected, candidates

    def _split_ideals(self) -> None:
        rng = random.Random(self.cfg.seed)
        for name, dirname in IDEAL_DIRS.items():
            if name == "Twinned(2 x 1)" and not self.cfg.include_twinned: continue
            files = sorted([p for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg") for p in (self.data_root / dirname).glob(ext)])
            if len(files) < 2: print(f"WARNING: {name} has fewer than 2 ideal images; excluded."); continue
            rng.shuffle(files)
            ntest = max(1, int(round(len(files) * self.cfg.test_fraction)))
            nutility = max(1, int(round(len(files) * self.cfg.utility_validation_fraction)))
            # The utility split trains/tunes acquisition only. Keep one or more
            # disjoint references even for small classes.
            ntest = min(ntest, len(files) - 2)
            nutility = min(nutility, len(files) - ntest - 1)
            if ntest < 1 or nutility < 1: print(f"WARNING: {name} lacks images for outer-test, utility-validation, and references; excluded."); continue
            self.test_images[name] = files[:ntest]
            self.utility_images[name] = files[ntest:ntest + nutility]
            self.references[name] = files[ntest + nutility:]
        if len(self.test_images) < 2: raise ValueError("Need at least two ideal reconstruction classes for evaluation.")

    def _load_bad_references(self) -> None:
        """Load absolute-scoring Bad images as negative anchors, never as ideal tests."""
        frame = pd.read_csv(self.absolute_csv)
        label_column = next((c for c in ("Reconstruction", "Reconstruction_Type", "Quality") if c in frame), None)
        path_column = next((c for c in ("Image_Path", "File_Path", "Path", "image_path") if c in frame), None)
        if label_column is None or path_column is None:
            print(f"WARNING: {self.absolute_csv.name} has no recognised Bad-label/path columns; bad anchors disabled.")
            return
        bad = frame[frame[label_column].astype(str).str.contains("bad", case=False, na=False)]
        for raw in bad[path_column].dropna().astype(str):
            path = (self.data_root / raw).resolve()
            if path.is_file(): self.bad_paths.append(path)

    def _write_manifests(self, initial: list[str], candidates: list[str]) -> None:
        base = Path(self.cfg.manifest_dir) if self.cfg.manifest_dir else self.output / "manifests"; paths = {
            "pretrain": base / "pairwise_for_pretrain" / "pretrain_pairs.csv",
            "candidate": base / "pairwise_for_candidate_pool" / "candidate_pairs.csv",
            "ideal": base / "ideal_for_test" / "ideal_test.csv"}
        for p in paths.values(): p.parent.mkdir(parents=True, exist_ok=True)
        def export(ids: list[str], path: Path):
            pd.concat([self.groups[i] for i in ids], ignore_index=True).to_csv(path, index=False)
        export(initial, paths["pretrain"]); export(candidates, paths["candidate"])
        rows = [{"class": c, "path": str(p.resolve()), "split": "test"} for c, ps in self.test_images.items() for p in ps]
        rows += [{"class": c, "path": str(p.resolve()), "split": "utility_validation"} for c, ps in self.utility_images.items() for p in ps]
        rows += [{"class": c, "path": str(p.resolve()), "split": "reference"} for c, ps in self.references.items() for p in ps]
        pd.DataFrame(rows).to_csv(paths["ideal"], index=False)
        audit = self.protocol_audit(initial, candidates)
        (base / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def protocol_audit(self, initial: Iterable[str], candidates: Iterable[str]) -> dict[str, Any]:
        initial, candidates = list(initial), list(candidates)
        reference = {p for ps in self.references.values() for p in ps}
        outer_test = {p for ps in self.test_images.values() for p in ps}
        utility = {p for ps in self.utility_images.values() for p in ps}
        pair_images = {p for pair_id in self.groups for p in self.groups[pair_id].resolved_img1.tolist() + self.groups[pair_id].resolved_img2.tolist()}
        return {
            "dataset_version": self.dataset.version,
            "input_hashes": file_hashes([self.pairwise_csv, self.absolute_csv, self.encoder_weights]),
            "ideal_image_hashes": file_hashes(reference | outer_test | utility),
            "initial_pair_count": len(initial), "candidate_pair_count": len(candidates),
            "exact_pair_overlap": len(set(initial) & set(candidates)),
            "image_overlap_initial_candidate": len(self._image_overlap(initial, candidates)),
            "reference_test_overlap": len(reference & outer_test), "utility_test_overlap": len(utility & outer_test),
            "reference_utility_overlap": len(reference & utility),
            "pairwise_image_overlap_outer_test": len(pair_images & outer_test),
            "bad_reference_count": len(self.bad_paths), "candidate_labels_hidden_from_selector": True,
            "candidate_oracle_is_pair_level": True,
        }

    def _image_overlap(self, left: Iterable[str], right: Iterable[str]) -> set[str]:
        images = lambda ids: {p for i in ids for p in self.groups[i].resolved_img1.tolist() + self.groups[i].resolved_img2.tolist()}
        return images(left) & images(right)

    def make_model(self) -> BTModel:
        seed_everything(self.cfg.seed)
        return BTModel(self.encoder_weights, dropout_p=self.cfg.dropout_p, metadata_dim=self.metadata_dim,
                       encoder_initialization=self.cfg.encoder_initialization).to(self.device)

    def rows_for(self, ids: Iterable[str]) -> pd.DataFrame:
        values = [self.groups[x] for x in ids]
        return pd.concat(values, ignore_index=True) if values else pd.DataFrame()

    def train(self, pair_ids: list[str], max_optimizer_updates: int | None = None) -> tuple[BTModel, dict[str, float]]:
        started = time.monotonic(); model = self.make_model(); rows = self.rows_for(pair_ids)
        if rows.empty: raise ValueError("Cannot train with no labeled pairs.")
        loader = DataLoader(PairRows(rows, self.cfg.symmetry_mode, self.metadata_vector if self.metadata_dim else None), batch_size=self.cfg.train_batch_size, shuffle=True, num_workers=0)
        optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        model.train(); losses: list[float] = []
        optimizer_updates = 0
        for epoch in range(self.cfg.epochs):
            for batch in loader:
                a, b, typ, winner, weight = batch[:5]
                metadata_a = metadata_b = None
                if self.metadata_dim:
                    metadata_a, metadata_b = batch[5].to(self.device), batch[6].to(self.device)
                a, b, typ, weight = a.to(self.device), b.to(self.device), typ.to(self.device), weight.to(self.device)
                ra, rb = model(a, metadata_a), model(b, metadata_b); ix = torch.arange(len(typ), device=self.device); xa, xb = ra[ix, typ], rb[ix, typ]
                loss_terms = []
                w = list(winner)
                for label in ("1", "2", "tie", "not_apply"):
                    mask = torch.tensor([x == label for x in w], device=self.device)
                    if not mask.any(): continue
                    if label == "1": term = -F.logsigmoid(xa[mask] - xb[mask])
                    elif label == "2": term = -F.logsigmoid(xb[mask] - xa[mask])
                    elif label == "tie": term = (xa[mask] - xb[mask]).abs()
                    else: term = F.relu(xa[mask]) + F.relu(xb[mask])
                    loss_terms.append((term * weight[mask]).mean() * mask.float().mean())
                loss = sum(loss_terms) if loss_terms else torch.tensor(0., device=self.device)
                # Fixed reference anchors set a useful absolute scale for each
                # reward head; reference images never appear in the test split.
                anchor_types = list(self.references)
                if len(anchor_types) > 1:
                    preferred = random.choice(anchor_types)
                    other = random.choice([x for x in anchor_types if x != preferred])
                    p_path, o_path = random.choice(self.references[preferred]), random.choice(self.references[other])
                    tf = transform(self.cfg.symmetry_mode)
                    with Image.open(p_path) as pi, Image.open(o_path) as oi:
                        pi = tf(pi.convert("L")).unsqueeze(0).to(self.device)
                        oi = tf(oi.convert("L")).unsqueeze(0).to(self.device)
                    anchor_idx = TYPE_TO_INDEX[preferred]
                    p_meta = torch.tensor(self.metadata_vector(str(p_path)), device=self.device).unsqueeze(0) if self.metadata_dim else None
                    o_meta = torch.tensor(self.metadata_vector(str(o_path)), device=self.device).unsqueeze(0) if self.metadata_dim else None
                    loss = loss + .25 * -F.logsigmoid(model(pi, p_meta)[0, anchor_idx] - model(oi, o_meta)[0, anchor_idx])
                if self.bad_paths:
                    bad_path = random.choice(self.bad_paths)
                    with Image.open(bad_path) as bad_image:
                        bad_tensor = transform(self.cfg.symmetry_mode)(symmetry_image(bad_image, self.cfg.symmetry_mode)).unsqueeze(0).to(self.device)
                    bad_meta = torch.tensor(self.metadata_vector(str(bad_path)), device=self.device).unsqueeze(0) if self.metadata_dim else None
                    loss = loss + self.cfg.bad_anchor_weight * F.relu(model(bad_tensor, bad_meta) + 1.0).mean()
                optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step(); losses.append(float(loss.detach().cpu()))
                optimizer_updates += 1
                if max_optimizer_updates is not None and optimizer_updates >= max_optimizer_updates:
                    break
            if max_optimizer_updates is not None and optimizer_updates >= max_optimizer_updates:
                break
            if time.monotonic() - started > 60: print(f"  training checkpoint: epoch {epoch + 1}/{self.cfg.epochs}", flush=True); started = time.monotonic()
        return model.eval(), {"loss": float(np.mean(losses)) if losses else float("nan"), "pairwise_accuracy": self.pairwise_accuracy(model, rows), "optimizer_updates": optimizer_updates}

    @torch.no_grad()
    def _score_paths(self, model: BTModel, paths: list[Path]) -> np.ndarray:
        tf = transform(self.cfg.symmetry_mode); batches = []
        for p in paths:
            with Image.open(p) as im: batches.append(tf(symmetry_image(im, self.cfg.symmetry_mode)))
        if not batches: return np.empty((0, len(TYPE_ORDER)))
        metadata = None
        if self.metadata_dim:
            metadata = torch.tensor(np.stack([self.metadata_vector(str(p)) for p in paths]), dtype=torch.float32, device=self.device)
        return model(torch.stack(batches).to(self.device), metadata).cpu().numpy()

    def evaluate(self, model: BTModel, split: str = "outer_test") -> dict[str, Any]:
        if split not in {"outer_test", "utility_validation"}:
            raise ValueError("split must be 'outer_test' or 'utility_validation'")
        images = self.test_images if split == "outer_test" else self.utility_images
        refs = {c: self._score_paths(model, ps) for c, ps in self.references.items()}
        details, correct, total = {}, 0, 0
        for truth, paths in images.items():
            c_ok = 0
            for path, score in zip(paths, self._score_paths(model, paths)):
                winrates = {}
                for candidate in refs:
                    opponents = np.concatenate([s for other, s in refs.items() if other != candidate])
                    idx = TYPE_TO_INDEX[candidate]
                    winrates[candidate] = float(np.mean(1 / (1 + np.exp(-(score[idx] - opponents[:, idx])))))
                prediction = max(winrates, key=winrates.get); ok = prediction == truth
                correct += ok; c_ok += ok; total += 1
            details[truth] = {"correct": c_ok, "total": len(paths), "accuracy": c_ok / len(paths)}
        return {"split": split, "test_accuracy": correct / total if total else float("nan"), "test_correct": correct, "test_total": total, "by_class": details}

    def pairwise_accuracy(self, model: BTModel, rows: pd.DataFrame) -> float:
        if rows.empty: return float("nan")
        ds = PairRows(rows, self.cfg.symmetry_mode, self.metadata_vector if self.metadata_dim else None); correct = total = 0
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=32):
                a, b, typ, winner, _ = batch[:5]
                ma = mb = None
                if self.metadata_dim: ma, mb = batch[5].to(self.device), batch[6].to(self.device)
                ra, rb = model(a.to(self.device), ma), model(b.to(self.device), mb); ix = torch.arange(len(typ), device=self.device)
                x, y = ra[ix, typ.to(self.device)], rb[ix, typ.to(self.device)]
                for left, right, label in zip(x.cpu(), y.cpu(), winner):
                    total += 1
                    is_correct = (left > right if label == "1" else right > left if label == "2" else abs(left-right) < .5 if label == "tie" else left < 0 and right < 0)
                    correct += bool(is_correct.item())
        return correct / total if total else float("nan")

    def candidates_with_clusters(self, ids: list[str], model: BTModel) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        candidates = []
        for pair_id in ids:
            item = dict(self.candidate_metadata.get(pair_id, {}))
            if not item:
                row = self.groups[pair_id].iloc[0]
                item = {"pair_id": pair_id, "img1": row.resolved_img1, "img2": row.resolved_img2}
            row = self.groups[pair_id].iloc[0]
            if self.metadata_dim: item.update(metadata1=self.metadata_vector(row.resolved_img1).tolist(), metadata2=self.metadata_vector(row.resolved_img2).tolist())
            candidates.append(item)
        cache = build_embedding_cache(candidates, model, self.device, self.cfg.symmetry_mode)
        vectors = np.stack([(cache[c["img1"]].numpy() + cache[c["img2"]].numpy()) / 2 for c in candidates])
        k = min(self.cfg.clusters, len(candidates))
        labels = KMeans(n_clusters=k, random_state=self.cfg.seed, n_init="auto").fit_predict(vectors) if k > 1 else np.zeros(len(candidates), dtype=int)
        for item, label, vector in zip(candidates, labels, vectors): item.update(cluster1=int(label), utility_features=vector.tolist())
        return candidates, cache

    def controlled_utility(self, labeled: list[str], candidate: str) -> dict[str, Any]:
        payload = {"labeled": sorted(labeled), "candidate": candidate, "epochs": self.cfg.epochs, "seed": self.cfg.seed, "test": self.cfg.test_fraction}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.utility_cache: return {**self.utility_cache[key], "cache_hit": True}
        base_model, _ = self.train(labeled); base = self.evaluate(base_model, split="utility_validation")
        aug_model, _ = self.train(labeled + [candidate]); augmented = self.evaluate(aug_model, split="utility_validation")
        by_class_delta = {name: augmented["by_class"][name]["accuracy"] - base["by_class"][name]["accuracy"] for name in base["by_class"]}
        value = {"candidate_pair_id": candidate, "utility_split": "utility_validation", "baseline_accuracy": base["test_accuracy"], "augmented_accuracy": augmented["test_accuracy"], "delta_accuracy": augmented["test_accuracy"] - base["test_accuracy"], "by_class_delta": by_class_delta, "cache_hit": False}
        self.utility_cache[key] = value; self.utility_cache_path.write_text(json.dumps(self.utility_cache, indent=2)); return value

    def _advanced_scores(self, candidates: list[dict[str, Any]], model: BTModel, cache: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """Attach pair-level mixture and metadata novelty scores without using test data."""
        if not candidates: return []
        with torch.no_grad():
            left = torch.stack([cache[x["img1"]] for x in candidates]).to(self.device)
            right = torch.stack([cache[x["img2"]] for x in candidates]).to(self.device)
            if self.metadata_dim:
                left_meta = torch.tensor([x["metadata1"] for x in candidates], dtype=torch.float32, device=self.device)
                right_meta = torch.tensor([x["metadata2"] for x in candidates], dtype=torch.float32, device=self.device)
                rewards = ((model.reward_from_embeddings(left, left_meta) + model.reward_from_embeddings(right, right_meta)) / 2).cpu().numpy()
            else:
                rewards = model.reward_from_embeddings((left + right) / 2).cpu().numpy()
        mix = mixture_metrics(rewards)
        metadata_score = np.zeros(len(candidates), dtype=float)
        if self.metadata_model is not None and self.metadata is not None:
            by_path = self.metadata.set_index("metadata_key")
            values = []
            for item in candidates:
                rows = [by_path.loc[p, self.metadata_columns].to_numpy(float) for p in (item["img1"], item["img2"]) if p in by_path.index]
                values.append(np.mean(rows, axis=0) if rows else self.metadata_model.center)
            metadata_score = self.metadata_model.score(np.asarray(values))
        output = []
        for i, item in enumerate(candidates):
            output.append({**item, "mixture_entropy": float(mix["mixture_entropy"][i]), "top_two_margin": float(mix["top_two_margin"][i]), "mixture_score": float(mix["mixture_score"][i]), "metadata_ood": float(metadata_score[i]), "class_probabilities": mix["probabilities"][i].tolist()})
        return output

    def select(self, strategy: str, candidates: list[dict[str, Any]], model: BTModel, cache: dict[str, torch.Tensor], history: list[dict[str, Any]], budget: int | None = None, labeled_ids: list[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, float | None]:
        n = min(self.cfg.batch_size if budget is None else budget, len(candidates))
        mc_strategies = {"mc_dropout_probability_variance", "mc_dropout_mutual_information", "mc_dropout_reward_variance"}
        if strategy in mc_strategies:
            all_scores, elapsed = score_mc_dropout(candidates, model, self.device, cache, mc_samples=self.cfg.mc_samples, seed=self.cfg.seed)
            return select_mc_dropout(all_scores, strategy, n), all_scores, elapsed
        if strategy in {"uncertainty_diversity", "core_set"}:
            labeled_items = []
            for pair_id in labeled_ids or []:
                row = self.groups[pair_id].iloc[0]
                labeled_items.append({"pair_id": pair_id, "img1": row.resolved_img1, "img2": row.resolved_img2})
            full_cache = build_embedding_cache(candidates + labeled_items, model, self.device, self.cfg.symmetry_mode)
            enriched = [{**item, "pair_vector": cached_pair_vector(item, full_cache).tolist()} for item in candidates]
            labeled_vectors = np.stack([cached_pair_vector(item, full_cache) for item in labeled_items]) if labeled_items else np.empty((0, 512))
            if strategy == "core_set": return core_set_select(enriched, labeled_vectors, n), None, None
            scored = score_uncertainty(enriched, model, self.device, full_cache)
            return uncertainty_diversity_select(scored, labeled_vectors, n, self.cfg.diversity_lambda), None, None
        if strategy == "random": return random_sampling(candidates, model, n, self.device, self.cfg.seed, cache), None, None
        if strategy == "uncertainty": return uncertainty_sampling(candidates, model, n, self.device, self.cfg.seed, cache), None, None
        if strategy == "cluster_quota_uncertainty":
            return cluster_quota_uncertainty_sampling(candidates, model, n, self.device, self.cfg.seed, cache), None, None
        if strategy == "cluster_margin_pairwise":
            return cluster_margin_pairwise_sampling(candidates, model, n, self.device, self.cfg.seed, cache), None, None
        if strategy in {"mixture_only", "uncertainty_plus_mixture", "metadata_ood", "uncertainty_plus_metadata"}:
            advanced = self._advanced_scores(candidates, model, cache)
            if strategy == "mixture_only": return sorted(advanced, key=lambda x: x["mixture_score"], reverse=True)[:n], advanced, None
            if strategy == "metadata_ood":
                if self.metadata_model is None: raise ValueError("metadata_ood requires --metadata-csv")
                return sorted(advanced, key=lambda x: x["metadata_ood"], reverse=True)[:n], advanced, None
            uncertain = score_uncertainty(advanced, model, self.device, cache)
            component = "mixture_score" if strategy == "uncertainty_plus_mixture" else "metadata_ood"
            if component == "metadata_ood" and self.metadata_model is None: raise ValueError("uncertainty_plus_metadata requires --metadata-csv")
            weight = self.cfg.mixture_weight if component == "mixture_score" else self.cfg.metadata_weight
            uncertainty_values = normalized([x["uncertainty"] for x in uncertain])
            component_values = normalized([x[component] for x in advanced])
            for item, u, u_value, extra in zip(advanced, uncertain, uncertainty_values, component_values):
                item["uncertainty"] = u["uncertainty"]; item["combined_score"] = float((1 - weight) * u_value + weight * extra)
            return sorted(advanced, key=lambda x: x["combined_score"], reverse=True)[:n], advanced, None
        if strategy == "utility":
            if len(history) < self.cfg.utility_min_history:
                print("Utility predictor has insufficient measured utilities; falling back to uncertainty.", flush=True)
                return uncertainty_sampling(candidates, model, n, self.device, self.cfg.seed, cache), None, None
            x = np.array([h["features"] for h in history]); y = np.array([h["utility"] for h in history])
            reg = RandomForestRegressor(n_estimators=100, random_state=self.cfg.seed, min_samples_leaf=2).fit(x, y)
            scored = score_uncertainty(candidates, model, self.device, cache)
            for item in scored: item["predicted_utility"] = float(reg.predict([item["utility_features"]])[0])
            return sorted(scored, key=lambda x: x["predicted_utility"], reverse=True)[:n], None, None
        raise ValueError(f"Unknown strategy {strategy}")

    def run_strategy(self, strategy: str, initial: list[str], pool: list[str]) -> list[dict[str, Any]]:
        labeled, available, logs, history = list(initial), list(pool), [], []
        spent = 0; checkpoint_dir = self.output / "checkpoints" / strategy; checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.acquisition_mode == "single-shot":
            model, train_metrics = self.train(labeled); before = self.evaluate(model)
            candidates, cache = self.candidates_with_clusters(available, model)
            selected, mc_scores, mc_elapsed = self.select(strategy, candidates, model, cache, history, budget=min(self.cfg.budget, len(available)), labeled_ids=labeled)
            self._write_advanced_scores(strategy, 0, selected, mc_scores)
            ids = [x["pair_id"] for x in selected]; after_model, after_train = self.train(labeled + ids); after = self.evaluate(after_model)
            record = {"strategy": strategy, "round": 0, "acquisition_mode": "single-shot", "budget_before": 0, "labeled_pairs": len(labeled), "candidate_remaining": len(available), "train_loss": train_metrics["loss"], "train_pairwise_accuracy": train_metrics["pairwise_accuracy"], "test_accuracy": before["test_accuracy"], "post_test_accuracy": after["test_accuracy"], "batch_utility": after["test_accuracy"] - before["test_accuracy"], "post_train_pairwise_accuracy": after_train["pairwise_accuracy"], "test_by_class": before["by_class"], "post_test_by_class": after["by_class"], "selected_pair_ids": ids, "selected_clusters": [x["cluster1"] for x in selected], "cluster_coverage": len({x["cluster1"] for x in selected}), "utilities": [], "mc_samples": self.cfg.mc_samples if mc_scores else None, "dropout_p": self.cfg.dropout_p if mc_scores else None, "mc_scoring_seconds": mc_elapsed, "final": True}
            torch.save({"model": after_model.state_dict(), "labeled_pair_ids": labeled + ids, "config": asdict(self.cfg)}, checkpoint_dir / "single_shot_final.pth")
            return [record]
        while available and spent < self.cfg.budget:
            model, train_metrics = self.train(labeled); evaluation = self.evaluate(model)
            candidates, cache = self.candidates_with_clusters(available, model)
            selected, mc_scores, mc_elapsed = self.select(strategy, candidates, model, cache, history, labeled_ids=labeled)
            self._write_advanced_scores(strategy, len(logs), selected, mc_scores)
            selected = selected[:min(self.cfg.budget - spent, len(selected))]
            utilities = [self.controlled_utility(labeled, item["pair_id"]) for item in selected] if self.cfg.utility_per_pair else []
            for item, util in zip(selected, utilities): history.append({"features": item["utility_features"], "utility": util["delta_accuracy"]})
            ids = [x["pair_id"] for x in selected]; coverage = len({x["cluster1"] for x in selected})
            selected_scores = [{key: item[key] for key in ("pair_id", "mc_probability_variance", "mc_mutual_information", "mc_reward_variance") if key in item} for item in selected]
            record = {"strategy": strategy, "round": len(logs), "budget_before": spent, "labeled_pairs": len(labeled), "candidate_remaining": len(available), "train_loss": train_metrics["loss"], "train_pairwise_accuracy": train_metrics["pairwise_accuracy"], "test_accuracy": evaluation["test_accuracy"], "test_by_class": evaluation["by_class"], "selected_pair_ids": ids, "revealed_preference_rows": {pair_id: len(self.groups[pair_id]) for pair_id in ids}, "selected_clusters": [x["cluster1"] for x in selected], "cluster_coverage": coverage, "utilities": utilities, "selected_mc_scores": selected_scores, "mc_samples": self.cfg.mc_samples if mc_scores else None, "dropout_p": self.cfg.dropout_p if mc_scores else None, "mc_scoring_seconds": mc_elapsed}
            if mc_scores is not None:
                score_rows = [{"strategy": strategy, "round": len(logs), "selected": item["pair_id"] in set(ids), "dropout_p": self.cfg.dropout_p, "mc_samples": self.cfg.mc_samples, "mc_scoring_seconds": mc_elapsed, **{key: item[key] for key in ("pair_id", "mc_probability_variance", "mc_mutual_information", "mc_reward_variance")}} for item in mc_scores]
                score_path = self.output / "mc_dropout_scores.csv"
                pd.DataFrame(score_rows).to_csv(score_path, mode="a", index=False, header=not score_path.exists())
            logs.append(record); torch.save({"model": model.state_dict(), "labeled_pair_ids": labeled, "config": asdict(self.cfg)}, checkpoint_dir / f"round_{len(logs):02d}.pth")
            labeled.extend(ids); available = [x for x in available if x not in set(ids)]; spent += len(ids)
            print(f"{strategy}: round {len(logs)} acquired {len(ids)}, budget={spent}, test accuracy={evaluation['test_accuracy']:.3f}, coverage={coverage}", flush=True)
        final_model, final_train = self.train(labeled); final_eval = self.evaluate(final_model)
        logs.append({"strategy": strategy, "round": len(logs), "budget_before": spent, "labeled_pairs": len(labeled), "candidate_remaining": len(available), "train_loss": final_train["loss"], "train_pairwise_accuracy": final_train["pairwise_accuracy"], "test_accuracy": final_eval["test_accuracy"], "test_by_class": final_eval["by_class"], "selected_pair_ids": [], "selected_clusters": [], "cluster_coverage": 0, "utilities": [], "final": True})
        return logs

    def _write_advanced_scores(self, strategy: str, round_number: int, selected: list[dict[str, Any]], scores: list[dict[str, Any]] | None) -> None:
        if not scores or not any("mixture_score" in row or "metadata_ood" in row for row in scores): return
        selected_ids = {x["pair_id"] for x in selected}
        rows = [{"strategy": strategy, "round": round_number, "selected": row["pair_id"] in selected_ids, **{key: value for key, value in row.items() if key not in {"utility_features"}}} for row in scores]
        path = self.output / "advanced_candidate_scores.csv"
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())

    def run(self) -> None:
        initial, pool = self.load_and_split(); all_logs = []
        for strategy in [s.strip() for s in self.cfg.strategies.split(",") if s.strip()]: all_logs.extend(self.run_strategy(strategy, initial, pool))
        (self.output / "experiment_log.json").write_text(json.dumps(all_logs, indent=2))
        flat = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in all_logs]
        pd.DataFrame(flat).to_csv(self.output / "experiment_log.csv", index=False)
        self._plot(all_logs); self._summary(all_logs); (self.output / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    def _plot(self, logs: list[dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sorted({x["strategy"] for x in logs}):
            rows = [x for x in logs if x["strategy"] == name]
            ax.plot([x["budget_before"] for x in rows], [x["test_accuracy"] for x in rows], marker="o", label=name)
        ax.set(xlabel="Acquired unique pairs", ylabel="Fixed ideal-test accuracy", title="Active learning downstream accuracy"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(self.output / "accuracy_vs_budget.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sorted({x["strategy"] for x in logs}):
            rows = [x for x in logs if x["strategy"] == name and x["utilities"]]
            if rows:
                ax.plot([x["budget_before"] for x in rows], [sum(u["delta_accuracy"] for u in x["utilities"]) for x in rows], marker="o", label=name)
        ax.set(xlabel="Acquired unique pairs", ylabel="Batch controlled utility", title="Measured acquisition utility")
        if ax.lines: ax.legend()
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(self.output / "utility_vs_budget.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sorted({x["strategy"] for x in logs}):
            rows = [x for x in logs if x["strategy"] == name and not x.get("final")]
            if rows: ax.plot([x["budget_before"] for x in rows], [x["cluster_coverage"] for x in rows], marker="o", label=name)
        ax.set(xlabel="Acquired unique pairs", ylabel="Clusters represented in batch", title="Selected-batch cluster coverage"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(self.output / "cluster_coverage_vs_budget.png", dpi=160); plt.close(fig)

    def _summary(self, logs: list[dict[str, Any]]) -> None:
        rows = []
        for strategy in sorted({x["strategy"] for x in logs}):
            trial = [x for x in logs if x["strategy"] == strategy]; final = trial[-1]
            active = [x["cluster_coverage"] for x in trial if not x.get("final")]
            rows.append({"strategy": strategy, "final_test_accuracy": final["test_accuracy"], "final_labeled_pairs": final["labeled_pairs"], "cumulative_measured_utility": sum(u["delta_accuracy"] for x in trial for u in x["utilities"]), "mean_batch_cluster_coverage": float(np.mean(active)) if active else 0.0})
        pd.DataFrame(rows).to_csv(self.output / "summary.csv", index=False)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    for name, typ, default in [("initial-pairs", int, 70), ("candidate-pairs", int, 100), ("budget", int, 50), ("batch-size", int, 5), ("epochs", int, 10), ("train-batch-size", int, 16), ("lr", float, 1e-4), ("test-fraction", float, .2), ("clusters", int, 20), ("seed", int, 42)]: p.add_argument(f"--{name}", type=typ, default=default)
    p.add_argument("--strategies", default="random,uncertainty,cluster_quota_uncertainty"); p.add_argument("--device", default="auto"); p.add_argument("--include-twinned", action="store_true"); p.add_argument("--no-utility-per-pair", dest="utility_per_pair", action="store_false"); p.add_argument("--utility-min-history", type=int, default=10); p.add_argument("--dropout-p", type=float, default=.2); p.add_argument("--mc-samples", type=int, default=20); p.add_argument("--acquisition-mode", choices=["sequential", "single-shot"], default="sequential"); p.add_argument("--diversity-lambda", type=float, default=.5); p.add_argument("--symmetry-mode", choices=SYMMETRY_MODES, default="none"); p.add_argument("--metadata-csv"); p.add_argument("--metadata-weight", type=float, default=.5); p.add_argument("--mixture-weight", type=float, default=.5); p.add_argument("--data-root"); p.add_argument("--dataset-version", choices=["v1.8", "v5.7"], default="v1.8"); p.add_argument("--utility-validation-fraction", type=float, default=.2); p.add_argument("--bad-anchor-weight", type=float, default=.10); p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args(); cfg = Config(**{k.replace("_", "-").replace("-", "_"): v for k, v in vars(a).items() if k != "smoke_test"})
    if a.smoke_test: cfg.initial_pairs, cfg.candidate_pairs, cfg.budget, cfg.batch_size, cfg.epochs, cfg.strategies = 8, 8, 2, 1, 1, "random"
    return cfg


if __name__ == "__main__":
    Experiment(parse_args()).run()

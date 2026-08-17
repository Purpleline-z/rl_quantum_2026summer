"""Optional, leakage-safe extensions for deployment-oriented RHEED experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps


SYMMETRY_MODES = ("none", "left_half_mirror", "symmetric_average")


def symmetry_image(image: Image.Image, mode: str = "none") -> Image.Image:
    """Return an in-memory symmetry ablation image; source files are untouched."""
    if mode not in SYMMETRY_MODES:
        raise ValueError(f"Unknown symmetry mode {mode!r}; choose from {SYMMETRY_MODES}")
    image = image.convert("L")
    if mode == "none":
        return image
    width, height = image.size
    left = image.crop((0, 0, max(1, width // 2), height))
    if mode == "left_half_mirror":
        return _mirror_full(left, width, height)
    arr = np.asarray(image, dtype=np.float32)
    return Image.fromarray(np.rint((arr + np.fliplr(arr)) / 2).astype(np.uint8), "L")


def _mirror_full(left: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("L", (width, height))
    result.paste(left, (0, 0))
    result.paste(ImageOps.mirror(left).resize((width - left.width, height)), (left.width, 0))
    return result


def apply_symmetry_tensor(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Tensor variant retained for tests and custom transforms."""
    if mode == "none": return tensor
    if mode == "left_half_mirror":
        half = tensor[..., : max(1, tensor.shape[-1] // 2)]
        return torch.cat((half, torch.flip(half, dims=(-1,))), dim=-1)[..., :tensor.shape[-1]]
    if mode == "symmetric_average": return (tensor + torch.flip(tensor, dims=(-1,))) / 2
    raise ValueError(mode)


def normalized(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if not len(values): return values
    lo, hi = np.nanmin(values), np.nanmax(values)
    return np.zeros_like(values) if not np.isfinite(hi - lo) or hi == lo else (values - lo) / (hi - lo)


def mixture_metrics(rewards: np.ndarray) -> dict[str, np.ndarray]:
    """Softmax evidence over reconstruction heads and uncertainty-of-type measures."""
    rewards = np.asarray(rewards, dtype=float)
    shifted = rewards - rewards.max(axis=1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(axis=1, keepdims=True)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(axis=1) / np.log(probability.shape[1])
    ordered = np.sort(probability, axis=1)
    return {"probabilities": probability, "mixture_entropy": entropy, "top_two_margin": ordered[:, -1] - ordered[:, -2], "mixture_score": entropy * (1 - (ordered[:, -1] - ordered[:, -2]))}


@dataclass
class MetadataModel:
    columns: list[str]
    center: np.ndarray
    scale: np.ndarray

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.sqrt(np.square((values - self.center) / self.scale).mean(axis=1))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Median-impute then standardize values using initial-labelled data only.

        Each numerical feature is paired with a missingness indicator.  This keeps
        missing process measurements explicit instead of turning them into a false
        physical zero.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.center[None, :], values)
        return np.concatenate(((filled - self.center[None, :]) / self.scale[None, :], missing.astype(float)), axis=1)


def load_metadata(path: str | Path, known_paths: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    key = next((x for x in ("path", "image_path", "image_id") if x in frame), None)
    if key is None: raise ValueError("Metadata CSV needs one of: path, image_path, image_id")
    frame = frame.rename(columns={key: "metadata_key"}).copy()
    # The contract is a resolved, normalized path.  Resolving both sides avoids
    # silent failures when a CSV uses a different slash convention.
    frame["metadata_key"] = frame.metadata_key.astype(str).map(lambda p: str(Path(p).resolve()).replace("\\", "/"))
    allowed = {str(Path(p).resolve()).replace("\\", "/") for p in known_paths}
    matched = frame[frame.metadata_key.isin(allowed)].copy()
    if matched.empty: raise ValueError("Metadata did not match any known image paths.")
    if matched.metadata_key.duplicated().any(): raise ValueError("Metadata contains duplicate image keys.")
    return matched


def fit_metadata_model(frame: pd.DataFrame, columns: list[str]) -> MetadataModel:
    data = frame[columns].apply(pd.to_numeric, errors="coerce")
    center = data.median().fillna(0).to_numpy(float)
    filled = data.fillna(pd.Series(center, index=columns)).to_numpy(float)
    scale = np.nanstd(filled, axis=0); scale[scale < 1e-8] = 1.0
    return MetadataModel(columns, center, scale)


def metadata_vector_for_path(frame: pd.DataFrame, model: MetadataModel, path: str) -> np.ndarray:
    """Return a stable fused-model vector, including all-missing fallback values."""
    rows = frame.loc[frame.metadata_key == str(Path(path).resolve()).replace("\\", "/"), model.columns]
    if rows.empty:
        raw = np.full((1, len(model.columns)), np.nan, dtype=float)
    else:
        raw = rows.iloc[[0]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return model.transform(raw)[0]

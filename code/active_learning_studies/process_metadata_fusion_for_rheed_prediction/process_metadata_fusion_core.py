"""Leakage-safe ingestion and model definitions for RHEED process metadata.

This module deliberately separates session-log parsing from experiment execution.
It can audit a single metadata-only session bundle, but training requires multiple
run-disjoint bundles and an explicit, verified map from laboratory frame paths to
image files already available to the project.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PRIMARY_FEATURE_COLUMNS = (
    "elapsed_s", "pyrometer_temp_C", "mistral_v_actual_V",
    "mistral_i_actual_A", "chamber_pressure_mbar",
    "pyrometer_temp_rate_C_per_s", "mistral_voltage_rate_V_per_s",
    "mistral_current_rate_A_per_s",
)
FORBIDDEN_METADATA_TERMS = (
    "recon_", "classifier", "grower_corrected", "live_label", "label",
    "note", "operation", "frame_quality", "event_state", "change_score",
)
REQUIRED_ARCHIVE_MEMBERS = (
    "session_metadata.json", "sensor_log.csv", "heartbeat_log.csv",
)
REQUIRED_SENSOR_COLUMNS = (
    "timestamp", "elapsed_s", "pyrometer_temp_C", "mistral_v_actual_V",
    "mistral_i_actual_A", "chamber_pressure_mbar",
)
REQUIRED_HEARTBEAT_COLUMNS = ("timestamp", "elapsed_s", "frame_path")


class MetadataSafetyError(ValueError):
    """Raised when a requested input could leak labels or cannot be audited."""


@dataclass(frozen=True)
class SessionBundle:
    archive_path: Path
    run_id: str
    metadata: dict[str, object]
    sensor_rows: pd.DataFrame
    heartbeat_rows: pd.DataFrame
    archive_sha256: str


@dataclass(frozen=True)
class TrainingOnlyMetadataScaler:
    columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.center[None, :], values)
        return np.concatenate(((filled - self.center[None, :]) / self.scale[None, :], missing.astype(np.float32)), axis=1).astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(bundle: zipfile.ZipFile, member: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(bundle.read(member)))
    except KeyError as exc:
        raise MetadataSafetyError(f"Archive lacks required member {member!r}.") from exc


def _parse_timestamp(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    value = frame.copy()
    value["timestamp"] = pd.to_datetime(value["timestamp"], errors="coerce")
    if value.timestamp.isna().any():
        raise MetadataSafetyError(f"{source} has invalid timestamp values.")
    return value.sort_values("timestamp", kind="stable").reset_index(drop=True)


def normalize_source_frame_path(value: object) -> str:
    """Normalize spelling only; this intentionally does not infer a new path."""
    return str(value).strip().replace("/", "\\").casefold()


def validate_requested_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(columns)
    if requested != PRIMARY_FEATURE_COLUMNS:
        extras = set(requested) - set(PRIMARY_FEATURE_COLUMNS)
        if extras:
            raise MetadataSafetyError(f"Only the registered primary features are allowed; forbidden/unregistered: {sorted(extras)}")
        raise MetadataSafetyError("All eight registered primary features are required in their documented order.")
    for name in requested:
        compact = name.casefold()
        if any(term in compact for term in FORBIDDEN_METADATA_TERMS):
            raise MetadataSafetyError(f"Label-like column {name!r} is not a metadata feature.")
    return requested


def read_session_bundle(archive_path: str | Path) -> SessionBundle:
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as bundle:
        names = set(bundle.namelist())
        missing = set(REQUIRED_ARCHIVE_MEMBERS) - names
        if missing:
            raise MetadataSafetyError(f"Archive {archive_path.name} lacks {sorted(missing)}")
        metadata = json.loads(bundle.read("session_metadata.json"))
        sensor = _read_csv(bundle, "sensor_log.csv")
        heartbeat = _read_csv(bundle, "heartbeat_log.csv")
    for name, frame, required in (("sensor_log.csv", sensor, REQUIRED_SENSOR_COLUMNS), ("heartbeat_log.csv", heartbeat, REQUIRED_HEARTBEAT_COLUMNS)):
        absent = set(required) - set(frame.columns)
        if absent:
            raise MetadataSafetyError(f"{name} lacks required columns {sorted(absent)}")
    sample_id, start = metadata.get("sample_id"), metadata.get("session_start")
    if not sample_id or not start:
        raise MetadataSafetyError("session_metadata.json needs non-empty sample_id and session_start.")
    sensor = _parse_timestamp(sensor, "sensor_log.csv")
    heartbeat = _parse_timestamp(heartbeat, "heartbeat_log.csv")
    run_id = f"{sample_id}__{str(start).replace(':', '').replace('-', '').replace('T', '_').replace('.', '_')}"
    return SessionBundle(archive_path, run_id, metadata, sensor, heartbeat, sha256_file(archive_path))


def _backward_rate(sensor: pd.DataFrame, value_column: str, output_column: str) -> pd.Series:
    values = pd.to_numeric(sensor[value_column], errors="coerce")
    seconds = sensor.timestamp.diff().dt.total_seconds()
    return values.diff() / seconds.where(seconds > 0)


def align_heartbeat_frames_causally(bundle: SessionBundle, frame_path_mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the latest prior sensor row to each heartbeat image reference."""
    sensor = bundle.sensor_rows.copy()
    sensor["pyrometer_temp_rate_C_per_s"] = _backward_rate(sensor, "pyrometer_temp_C", "pyrometer_temp_rate_C_per_s")
    sensor["mistral_voltage_rate_V_per_s"] = _backward_rate(sensor, "mistral_v_actual_V", "mistral_voltage_rate_V_per_s")
    sensor["mistral_current_rate_A_per_s"] = _backward_rate(sensor, "mistral_i_actual_A", "mistral_current_rate_A_per_s")
    sensor = sensor.rename(columns={"timestamp": "source_sensor_timestamp", "elapsed_s": "sensor_elapsed_s"})
    # Heartbeat logs may repeat a temperature convenience field.  The registered
    # process vector is sourced exclusively from the timestamped sensor log.
    frames = bundle.heartbeat_rows.loc[:, ["timestamp", "elapsed_s", "frame_path"]].rename(columns={"timestamp": "frame_timestamp", "elapsed_s": "frame_elapsed_s"}).copy()
    merged = pd.merge_asof(frames.sort_values("frame_timestamp"), sensor.sort_values("source_sensor_timestamp"), left_on="frame_timestamp", right_on="source_sensor_timestamp", direction="backward", allow_exact_matches=True)
    if (merged.source_sensor_timestamp.dropna() > merged.frame_timestamp.loc[merged.source_sensor_timestamp.notna()]).any():
        raise AssertionError("Causal join selected a future sensor row.")
    merged["sensor_lag_seconds"] = (merged.frame_timestamp - merged.source_sensor_timestamp).dt.total_seconds()
    # The registered elapsed-time feature is the image/frame clock, not the
    # timestamp of the preceding sensor sample.
    merged["elapsed_s"] = pd.to_numeric(merged["frame_elapsed_s"], errors="coerce")
    merged["run_id"] = bundle.run_id
    merged["archive_sha256"] = bundle.archive_sha256
    merged["source_frame_path"] = merged.frame_path.map(normalize_source_frame_path)
    merged["resolved_image_path"] = pd.NA
    if frame_path_mapping is not None:
        required = {"source_frame_path", "resolved_image_path"}
        absent = required - set(frame_path_mapping.columns)
        if absent:
            raise MetadataSafetyError(f"Frame-path mapping lacks {sorted(absent)}")
        mapping = frame_path_mapping.loc[:, ["source_frame_path", "resolved_image_path"]].copy()
        mapping["source_frame_path"] = mapping.source_frame_path.map(normalize_source_frame_path)
        if mapping.source_frame_path.duplicated().any():
            raise MetadataSafetyError("Frame-path mapping contains duplicate source_frame_path values.")
        merged = merged.drop(columns="resolved_image_path").merge(mapping, on="source_frame_path", how="left", validate="many_to_one")
        merged["resolved_image_path"] = merged.resolved_image_path.map(lambda value: str(Path(value).expanduser().resolve()) if pd.notna(value) else pd.NA)
        exists = merged.resolved_image_path.map(lambda value: pd.notna(value) and Path(value).is_file())
        merged.loc[~exists, "resolved_image_path"] = pd.NA
    validate_requested_feature_columns(PRIMARY_FEATURE_COLUMNS)
    merged["feature_missing_count"] = merged.loc[:, PRIMARY_FEATURE_COLUMNS].isna().sum(axis=1)
    return merged.sort_values("frame_timestamp", kind="stable").reset_index(drop=True)


def fit_training_only_scaler(training_rows: pd.DataFrame, columns: Iterable[str] = PRIMARY_FEATURE_COLUMNS) -> TrainingOnlyMetadataScaler:
    columns = validate_requested_feature_columns(columns)
    if training_rows.empty:
        raise MetadataSafetyError("Cannot fit metadata normalization without training rows.")
    values = training_rows.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    center = values.median().fillna(0.0).to_numpy(float)
    filled = values.fillna(pd.Series(center, index=columns)).to_numpy(float)
    scale = np.nanstd(filled, axis=0)
    scale[scale < 1e-8] = 1.0
    return TrainingOnlyMetadataScaler(columns, center, scale)


def build_run_disjoint_split(frame_manifest: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic run-level split or fail before any experiment starts."""
    required = {"run_id", "source_frame_path", "resolved_image_path"}
    absent = required - set(frame_manifest.columns)
    if absent:
        raise MetadataSafetyError(f"Frame manifest lacks {sorted(absent)}")
    if frame_manifest.resolved_image_path.isna().any():
        raise MetadataSafetyError("Cannot build experiment split while any frame path is unresolved.")
    runs = sorted(frame_manifest.run_id.astype(str).unique())
    if len(runs) < 3:
        raise MetadataSafetyError(f"Run-disjoint train/validation/outer-test requires at least 3 usable sessions; found {len(runs)}.")
    duplicate_across_runs = frame_manifest.groupby("resolved_image_path").run_id.nunique()
    if (duplicate_across_runs > 1).any():
        raise MetadataSafetyError("An image path occurs in multiple session runs; cannot prove image-disjointness.")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(runs))
    partitions = {order[0]: "outer_test", order[1]: "validation"}
    partitions.update({run: "train" for run in order[2:]})
    output = frame_manifest.copy()
    output["split"] = output.run_id.astype(str).map(partitions)
    by_split = output.groupby("split").resolved_image_path.apply(set)
    if by_split["train"] & by_split["validation"] or by_split["train"] & by_split["outer_test"] or by_split["validation"] & by_split["outer_test"]:
        raise AssertionError("Image-disjoint split verification failed.")
    return output


def preflight_summary(bundle: SessionBundle, aligned: pd.DataFrame) -> dict[str, object]:
    return {
        "archive_path": str(bundle.archive_path), "archive_sha256": bundle.archive_sha256,
        "run_id": bundle.run_id, "session_start": bundle.metadata.get("session_start"),
        "session_end": bundle.metadata.get("session_end"), "sensor_row_count": int(len(bundle.sensor_rows)),
        "heartbeat_frame_count": int(len(bundle.heartbeat_rows)), "archive_contains_image_bytes": False,
        "primary_features": list(PRIMARY_FEATURE_COLUMNS),
        "aligned_frame_count": int(len(aligned)), "unresolved_frame_count": int(aligned.resolved_image_path.isna().sum()),
        "future_sensor_match_count": int((aligned.source_sensor_timestamp > aligned.frame_timestamp).fillna(False).sum()),
        "sensor_lag_seconds": {"min": float(aligned.sensor_lag_seconds.min(skipna=True)) if aligned.sensor_lag_seconds.notna().any() else None,
                               "max": float(aligned.sensor_lag_seconds.max(skipna=True)) if aligned.sensor_lag_seconds.notna().any() else None},
        "training_eligible": False,
        "training_blocker": "A single metadata-only session cannot satisfy run-disjoint image-resolved train/validation/outer-test capacity.",
    }


def write_json(value: object, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _reward_head(hidden_dim: int, dropout_p: float, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_p), nn.Linear(hidden_dim, output_dim))


class MetadataOnlyBradleyTerryRewardModel(nn.Module):
    """Metadata-only ablation with the same five reward outputs as the image arm."""
    def __init__(self, metadata_dim: int = 16, hidden_dim: int = 256, dropout_p: float = 0.2, output_dim: int = 5):
        super().__init__()
        if metadata_dim != 16:
            raise ValueError("This registered study requires 16 metadata values: eight features plus eight missingness flags.")
        self.metadata_branch = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.to_reward_space = nn.Sequential(nn.Linear(64, 512), nn.ReLU(inplace=True))
        self.reward_head = _reward_head(hidden_dim, dropout_p, output_dim)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        if metadata.shape[-1] != 16:
            raise ValueError("Metadata-only model requires vectors with width 16.")
        return self.reward_head(self.to_reward_space(self.metadata_branch(metadata)))

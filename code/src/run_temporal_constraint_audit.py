#!/usr/bin/env python3
"""Descriptive audit of RHEED trajectory ordering before any temporal model is used.

This tool intentionally does not apply the tentative physics rules as labels, losses,
or selection constraints.  A prediction-based audit requires a saved, validated
downstream Bradley--Terry classifier; short (<=120 minute) jobs deliberately do not
write model checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from project_paths import RESULT_ROOT, resolve_data_root


FILENAME = re.compile(
    r"^(?P<index>\d+)_(?P<run>RR[^_]+)_(?P<temperature>\d+)C_(?P<frame>[^.]+)$",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frame(path: Path, trajectories_root: Path) -> dict:
    """Parse only filename-supported ordering; never infer a physical transition."""
    relative = path.relative_to(trajectories_root)
    date = relative.parts[0] if len(relative.parts) > 1 else "unknown_date"
    match = FILENAME.match(path.stem)
    if match:
        values = match.groupdict()
        return {
            "relative_image_path": relative.as_posix(), "date_folder": date,
            "trajectory_id": f"{date}/{values['run']}", "frame_index": int(values["index"]),
            "filename_frame_token": values["frame"], "temperature_c": int(values["temperature"]),
            "ordering_source": "leading_filename_index", "parse_status": "parsed",
        }
    return {
        "relative_image_path": relative.as_posix(), "date_folder": date,
        "trajectory_id": f"{date}/unparsed", "frame_index": None,
        "filename_frame_token": None, "temperature_c": None,
        "ordering_source": "unavailable", "parse_status": "unparsed_filename",
    }


def build_trajectory_manifest(trajectories_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(p for p in trajectories_root.rglob("*") if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    frames = pd.DataFrame([parse_frame(path, trajectories_root) for path in paths])
    if frames.empty:
        return frames, pd.DataFrame()
    frames = frames.sort_values(["trajectory_id", "frame_index", "relative_image_path"], na_position="last").reset_index(drop=True)
    summaries = []
    for trajectory_id, group in frames.groupby("trajectory_id", dropna=False):
        numeric = sorted(int(x) for x in group.frame_index.dropna())
        duplicate_indices = len(numeric) - len(set(numeric))
        missing_indices = sum(max(0, b - a - 1) for a, b in zip(numeric, numeric[1:]))
        summaries.append({
            "trajectory_id": trajectory_id, "frame_count": len(group),
            "parsed_frame_count": int(group.frame_index.notna().sum()),
            "unparsed_frame_count": int(group.frame_index.isna().sum()),
            "duplicate_frame_index_count": duplicate_indices, "missing_index_count": missing_indices,
            "first_relative_image_path": group.iloc[0].relative_image_path,
            "last_relative_image_path": group.iloc[-1].relative_image_path,
            "order_ambiguous": bool(group.frame_index.isna().any() or duplicate_indices),
        })
    return frames, pd.DataFrame(summaries)


def run_audit(data_root: str | Path | None = None, output_root: Path | None = None) -> Path:
    root = resolve_data_root(data_root)
    trajectories = root / "original data" / "Trajectories"
    constraints = root / "temporal_constraints.json"
    if not trajectories.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectories}")
    if not constraints.is_file():
        raise FileNotFoundError(f"Tentative-constraint configuration not found: {constraints}")
    output = output_root or (RESULT_ROOT / "temporal_constraint_audit")
    output.mkdir(parents=True, exist_ok=True)
    frames, summaries = build_trajectory_manifest(trajectories)
    frames.to_csv(output / "trajectory_frame_manifest.csv", index=False)
    summaries.to_csv(output / "trajectory_order_summary.csv", index=False)
    config = json.loads(constraints.read_text(encoding="utf-8"))
    audit = {
        "input_data_root": "data (resolved at runtime; reusable paths below are relative)",
        "trajectory_root_relative": "original data/Trajectories",
        "temporal_constraints_relative": "temporal_constraints.json",
        "temporal_constraints_sha256": sha256(constraints),
        "frame_count": int(len(frames)), "trajectory_count": int(len(summaries)),
        "tentative_rules": config, "prediction_scoring_completed": False,
        "prediction_scoring_reason": "No saved validated downstream classifier checkpoint is available. Default short jobs do not save model checkpoints.",
        "rule_application": "No tentative rule was applied to training, selection, or decoding.",
    }
    (output / "temporal_audit_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = """# Temporal Constraint Audit\n\nThis is a descriptive filename/order audit, not a temporal classifier experiment. The physics-team statements remain tentative configuration only; they were not converted into labels, a loss, a decoding rule, or a selection constraint.\n\n## What is available now\n\n- `trajectory_frame_manifest.csv` gives each frame's parsed trajectory ID and leading filename index.\n- `trajectory_order_summary.csv` reports missing indices, duplicate indices, unparsed names, and order ambiguity. A numeric gap indicates missing observed filenames, not a missing physical state.\n\n## What is deliberately not claimed\n\nNo model-based frequency is reported for “starts at 1x1,” “HTR comes last,” or “1x1 passes through bad.” A frozen SimCLR encoder is a representation, not a trained reconstruction-type classifier, and the short-run checkpoint policy leaves no saved downstream classifier to score frames. In particular, the current classifier setup has no validated `bad` state detector, so the third rule cannot be evaluated honestly from its outputs.\n\nWhen the physics team confirms rule applicability and a saved, validated downstream classifier is available, a separate prediction audit can score trajectories. That future audit must report ambiguous ordering and violations, while keeping metadata fusion separate from whole-trajectory constraints.\n"""
    (output / "temporal_audit_report.md").write_text(report, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Stable data directory; defaults to RHEED_DATA_ROOT or sibling data/.")
    parser.add_argument("--output-root", type=Path, help="Override local generated-results directory.")
    args = parser.parse_args()
    print(run_audit(args.data_root, args.output_root))


if __name__ == "__main__":
    main()

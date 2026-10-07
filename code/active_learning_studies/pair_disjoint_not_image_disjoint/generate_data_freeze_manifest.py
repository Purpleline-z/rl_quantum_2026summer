"""Generate a reproducibility / data-freeze manifest for a single seed.

Usage:
    python generate_data_freeze_manifest.py \
        --settings literature_backed_pair_disjoint_budget_curve_settings.json \
        --seed 42 \
        --data-root /path/to/data \
        [--out manifest_seed42.json]

The script fails with a non-zero exit code if:
  - any ZERO_AUDIT_FIELDS field is non-zero,
  - Twinned(2 x 1) appears in the final pairwise groups or ideal splits,
  - expected partition sizes differ from those recorded at freeze time (only
    when --expected-sizes is supplied).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAM = HERE.parent.parent / "active_learning_program"
sys.path.insert(0, str(PROGRAM))

from pairwise_active_learning_pipeline import Config, Experiment, TYPE_ORDER

ZERO_AUDIT_FIELDS = (
    "exact_pair_overlap",
    "reference_test_identity_overlap",
    "utility_test_identity_overlap",
    "reference_utility_identity_overlap",
    "pairwise_image_identity_overlap_outer_test",
    "pairwise_image_identity_overlap_reference",
    "pairwise_image_identity_overlap_utility_validation",
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _paths_by_split(exp: Experiment) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"reference": [], "utility_validation": [], "outer_test": []}
    for cls, paths in exp.references.items():
        result["reference"].extend(str(p) for p in paths)
    for cls, paths in exp.utility_images.items():
        result["utility_validation"].extend(str(p) for p in paths)
    for cls, paths in exp.test_images.items():
        result["outer_test"].extend(str(p) for p in paths)
    return {k: sorted(v) for k, v in result.items()}


def _counts_by_class(split_dict: dict) -> dict[str, int]:
    return {cls: len(paths) for cls, paths in sorted(split_dict.items())}


def generate(settings_path: Path, seed: int, data_root: str,
             out_path: Path | None, expected_sizes: dict | None) -> dict:
    settings = json.loads(settings_path.read_text())
    cfg = Config(
        data_root=data_root,
        seed=seed,
        initial_pairs=settings.get("initial_pair_groups", 10),
        candidate_pairs=settings.get("maximum_acquired_pair_groups", 100),
        dataset_version=settings.get("dataset_version", "v1.8"),
        include_twinned=False,
    )
    exp = Experiment(cfg)
    initial, pool = exp.load_and_split()
    audit = exp.protocol_audit(initial, pool)

    # ── Hard failures ────────────────────────────────────────────────────────
    failures: list[str] = []

    for field in ZERO_AUDIT_FIELDS:
        if audit.get(field, 0) != 0:
            failures.append(f"VIOLATION: {field} = {audit[field]} (must be 0)")

    twinned_in_pairwise = any(
        "Twinned(2 x 1)" in set(grp.canonical_type)
        for grp in exp.groups.values()
    )
    if twinned_in_pairwise:
        failures.append("VIOLATION: Twinned(2 x 1) found in pairwise groups")

    for split_name, split_dict in [
        ("test_images", exp.test_images),
        ("utility_images", exp.utility_images),
        ("references", exp.references),
    ]:
        if "Twinned(2 x 1)" in split_dict:
            failures.append(f"VIOLATION: Twinned(2 x 1) found in {split_name}")

    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        sys.exit(1)

    # ── Build manifest ───────────────────────────────────────────────────────
    ideal_paths_by_split = _paths_by_split(exp)

    pair_group_assignments: dict[str, str] = {}
    for pair_id in initial:
        pair_group_assignments[pair_id] = "initial_labelled"
    for pair_id in pool:
        pair_group_assignments[pair_id] = "candidate_pool"

    pairwise_csv = exp.pairwise_csv
    absolute_csv = exp.absolute_csv

    included_classes = [t for t in TYPE_ORDER if t != "Twinned(2 x 1)"]
    excluded_classes = ["Twinned(2 x 1)"]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_hash(),
        "settings_file": str(settings_path.resolve()),
        "seed": seed,
        "dataset_version": cfg.dataset_version,
        "included_reconstruction_classes": included_classes,
        "excluded_reconstruction_classes": excluded_classes,
        "source_files": {
            "pairwise_csv": str(pairwise_csv),
            "absolute_csv": str(absolute_csv),
            "pairwise_csv_sha256": _sha256(Path(pairwise_csv)),
            "absolute_csv_sha256": _sha256(Path(absolute_csv)),
        },
        "partition_counts": {
            "initial_labelled_pair_groups": len(initial),
            "candidate_pool_pair_groups": len(pool),
            "reference_images": sum(len(v) for v in exp.references.values()),
            "utility_validation_images": sum(len(v) for v in exp.utility_images.values()),
            "outer_test_images": sum(len(v) for v in exp.test_images.values()),
        },
        "counts_by_class": {
            "reference": _counts_by_class(exp.references),
            "utility_validation": _counts_by_class(exp.utility_images),
            "outer_test": _counts_by_class(exp.test_images),
        },
        "initial_labelled_pair_ids": sorted(initial),
        "candidate_pool_pair_ids": sorted(pool),
        "pair_group_assignments": pair_group_assignments,
        "ideal_image_paths_by_split": ideal_paths_by_split,
        "forbidden_overlap_audit": {f: audit[f] for f in ZERO_AUDIT_FIELDS},
        "full_protocol_audit": audit,
    }

    # ── Expected-size guard ──────────────────────────────────────────────────
    if expected_sizes:
        size_failures: list[str] = []
        for key, expected in expected_sizes.items():
            actual = manifest["partition_counts"].get(key)
            if actual != expected:
                size_failures.append(
                    f"SIZE MISMATCH: {key} expected {expected}, got {actual}"
                )
        if size_failures:
            for msg in size_failures:
                print(msg, file=sys.stderr)
            sys.exit(1)

    # ── Write output ─────────────────────────────────────────────────────────
    if out_path is None:
        out_path = HERE / f"data_freeze_manifest_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Human-readable summary
    summary_lines = [
        f"Data Freeze Manifest — seed {seed}",
        f"  Generated : {manifest['generated_utc']}",
        f"  Git commit: {manifest['git_commit_hash']}",
        f"  Dataset   : {cfg.dataset_version}",
        f"  Included classes: {', '.join(included_classes)}",
        f"  Excluded classes: {', '.join(excluded_classes)}",
        "",
        "Partition counts:",
        f"  Initial labelled pair groups : {manifest['partition_counts']['initial_labelled_pair_groups']}",
        f"  Candidate pool pair groups   : {manifest['partition_counts']['candidate_pool_pair_groups']}",
        f"  Reference images             : {manifest['partition_counts']['reference_images']}",
        f"  Utility-validation images    : {manifest['partition_counts']['utility_validation_images']}",
        f"  Outer-test images            : {manifest['partition_counts']['outer_test_images']}",
        "",
        "Forbidden-overlap audit (all must be 0):",
    ]
    for field in ZERO_AUDIT_FIELDS:
        summary_lines.append(f"  {field}: {manifest['forbidden_overlap_audit'][field]}")

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = out_path.with_suffix(".summary.txt")
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Manifest written to {out_path}")
    print(f"Summary  written to {summary_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--expected-sizes", type=json.loads, default=None,
        help='JSON dict of partition_counts keys → expected values, '
             'e.g. \'{"initial_labelled_pair_groups": 10}\'',
    )
    args = parser.parse_args()
    generate(args.settings, args.seed, args.data_root, args.out, args.expected_sizes)


if __name__ == "__main__":
    main()

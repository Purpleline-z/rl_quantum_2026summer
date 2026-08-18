#!/usr/bin/env python3
"""Export the default identity-safe ideal-image split capacity.

This mirrors Experiment._split_ideals in the active-learning pipeline:
SHA-256 identities are globally unique, the default excludes Twinned(2 x 1),
and each remaining class is split into 20% outer test, 20% utility validation,
and the remaining reference anchors.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


STUDY = Path(__file__).resolve().parent
DATA_ROOT = STUDY.parents[2] / "data" / "original data"
OUTPUT = STUDY / "results" / "identity_safe_protocol" / "ideal_split_capacity.csv"
IDEAL_DIRS = {
    "(1 x 1)": "STO_ideal_1x1",
    "c(6 x 2)": "STO_ideal_c6x2",
    "(√13 x √13)": "STO_ideal_RT13",
    "HTR": "STO_ideal_HTR",
}
EXTENSIONS = ("*.png", "*.bmp", "*.jpg", "*.jpeg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assigned: set[str] = set()
    rows: list[dict[str, int | str]] = []
    for label, dirname in IDEAL_DIRS.items():
        files = sorted(path for pattern in EXTENSIONS for path in (DATA_ROOT / dirname).glob(pattern))
        unique = []
        for path in files:
            identity = sha256(path)
            if identity not in assigned:
                assigned.add(identity)
                unique.append(path)
        if len(unique) < 3:
            raise ValueError(f"{label} has fewer than three unique ideal images in {DATA_ROOT / dirname}.")
        n_test = min(max(1, round(len(unique) * 0.2)), len(unique) - 2)
        n_validation = min(max(1, round(len(unique) * 0.2)), len(unique) - n_test - 1)
        rows.append(
            {
                "class": label,
                "raw_files": len(files),
                "unique_identities": len(unique),
                "outer_test": n_test,
                "utility_validation": n_validation,
                "reference": len(unique) - n_test - n_validation,
            }
        )
    rows.append(
        {
            "class": "Total (default classes; Twinned excluded)",
            "raw_files": sum(int(row["raw_files"]) for row in rows),
            "unique_identities": sum(int(row["unique_identities"]) for row in rows),
            "outer_test": sum(int(row["outer_test"]) for row in rows),
            "utility_validation": sum(int(row["utility_validation"]) for row in rows),
            "reference": sum(int(row["reference"]) for row in rows),
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()

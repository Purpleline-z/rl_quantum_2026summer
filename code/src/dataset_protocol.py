"""Versioned, auditable input contracts for Stage-A active-learning studies."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    version: str
    pairwise_name: str
    absolute_name: str
    formal_classes: tuple[str, ...]


SPECS = {
    "v1.8": DatasetSpec("v1.8", "Quantum Label Data - Pairwise_Comparisonv1.8.csv",
                         "Quantum Label Data - Absolute_Scoringv1.8 (1).csv",
                         ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")),
    "v5.7": DatasetSpec("v5.7", "Quantum Label Data - Pairwise_Comparisonv5.7.csv",
                         "Quantum Label Data - Absolute_Scoringv5.7.csv",
                         ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")),
}


def dataset_spec(version: str) -> DatasetSpec:
    try:
        return SPECS[version]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset version {version!r}; choose one of {sorted(SPECS)}.") from exc


def locate_input(data_root: Path, code_root: Path, name: str) -> Path:
    """Find a tracked label export without silently falling back to another version."""
    candidates = (data_root / "original data" / name, data_root / name,
                  code_root / "classifier2" / name, code_root / "Classifier2" / name)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    rendered = "\n  ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Required Stage-A input {name!r} was not found. Looked in:\n  {rendered}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in paths if path.is_file()}

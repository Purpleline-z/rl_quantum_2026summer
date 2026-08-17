"""Central path configuration for local and Google Colab RHEED runs."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PROJECT_ROOT = CODE_ROOT.parent
PAIR_DISJOINT_STUDY_ROOT = CODE_ROOT / "active_learning_studies" / "pair_disjoint_not_image_disjoint"
RESULT_ROOT = PAIR_DISJOINT_STUDY_ROOT / "results"
IMAGE_REPRESENTATION_RESULTS_ROOT = CODE_ROOT / "active_learning_studies" / "image_representation_analysis" / "results"
TRAJECTORY_ORDERING_RESULTS_ROOT = CODE_ROOT / "active_learning_studies" / "rheed_trajectory_ordering_analysis" / "results"


def resolve_data_root(override: str | Path | None = None) -> Path:
    """Return stable input data, allowing an environment or CLI override."""
    root = Path(override or os.environ.get("RHEED_DATA_ROOT") or PROJECT_ROOT / "data").expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RHEED data directory does not exist: {root}. Set RHEED_DATA_ROOT or pass --data-root.")
    return root

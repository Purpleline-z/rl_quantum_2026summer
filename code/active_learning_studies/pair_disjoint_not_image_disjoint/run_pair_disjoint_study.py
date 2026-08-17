#!/usr/bin/env python3
"""Launch the existing pair-disjoint, not image-disjoint selector comparison."""
from __future__ import annotations

import sys
from pathlib import Path

PROGRAM_ROOT = Path(__file__).resolve().parents[2] / "active_learning_program"
sys.path.insert(0, str(PROGRAM_ROOT))

from compare_five_pair_selection_methods import main


if __name__ == "__main__":
    main()

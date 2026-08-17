"""Shared path contract for the preserved Classifier2 scripts."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPTS_ROOT.parent.parent
sys.path.insert(0, str(CODE_ROOT / "active_learning_program"))
from project_file_locations import RESULT_ROOT, resolve_data_root

DATA_ROOT = resolve_data_root() / "original data"
CLASSIFIER2_ROOT = CODE_ROOT / "classifier2"
OUTPUT_ROOT = RESULT_ROOT / "classifier2_legacy"
PAIRWISE_CSV = DATA_ROOT / "Quantum Label Data - Pairwise_Comparisonv1.8.csv"
ABSOLUTE_CSV = DATA_ROOT / "Quantum Label Data - Absolute_Scoringv1.8 (1).csv"
PRETRAINED_ENCODER = CLASSIFIER2_ROOT / "simclr_encoder_checkpoint" / "simclr_resnet18_encoder.pth"

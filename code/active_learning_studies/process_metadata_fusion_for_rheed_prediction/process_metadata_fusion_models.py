"""Registered image-only, metadata-only, and late-fusion reward-model arms."""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROGRAM_ROOT = HERE.parents[1] / "active_learning_program"
sys.path.insert(0, str(PROGRAM_ROOT))

from pairwise_active_learning_pipeline import BTModel, TYPE_ORDER
from process_metadata_fusion_core import MetadataOnlyBradleyTerryRewardModel


def build_registered_reward_model(arm: str, simclr_checkpoint: Path | None, dropout_p: float = 0.2) -> nn.Module:
    """Build one registered architecture without starting a training experiment."""
    if arm == "image_only":
        return BTModel(simclr_checkpoint, dropout_p=dropout_p, metadata_dim=0, encoder_initialization="simclr")
    if arm == "metadata_only":
        return MetadataOnlyBradleyTerryRewardModel(metadata_dim=16, dropout_p=dropout_p, output_dim=len(TYPE_ORDER))
    if arm == "image_plus_process_metadata_late_fusion":
        return BTModel(simclr_checkpoint, dropout_p=dropout_p, metadata_dim=16, encoder_initialization="simclr")
    raise ValueError(f"Unknown registered arm {arm!r}.")


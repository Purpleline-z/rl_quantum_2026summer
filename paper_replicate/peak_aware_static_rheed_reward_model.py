"""Compact peak-aware image encoder for static RHEED preference learning."""
from __future__ import annotations

import torch
import hashlib
from torch import nn
from torch.nn import functional as F
from torchvision import models
from pathlib import Path

RECONSTRUCTION_TYPES = ("(1 x 1)", "Twinned(2 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")
_CHECKPOINT_HASHES: dict[str, str] = {}


def checkpoint_sha256(path: Path) -> str:
    """Hash the fixed laboratory checkpoint once per Python process for provenance."""
    key = str(path.resolve())
    if key not in _CHECKPOINT_HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        _CHECKPOINT_HASHES[key] = digest.hexdigest()
    return _CHECKPOINT_HASHES[key]


class RHEEDPeakFeatureExtractor(nn.Module):
    """Summarise horizontal/vertical diffraction intensity profiles without scipy."""
    output_dim = 24

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        horizontal = gray.mean(dim=2)  # [B, 1, W]
        vertical = gray.mean(dim=3)    # [B, 1, H]
        horizontal_bins = F.adaptive_avg_pool1d(horizontal, 8).squeeze(1)
        vertical_bins = F.adaptive_avg_pool1d(vertical, 8).squeeze(1)
        statistics = torch.cat((
            gray.mean((1, 2, 3), keepdim=True).flatten(1),
            gray.std((1, 2, 3), keepdim=True).flatten(1),
            gray.amax((1, 2, 3), keepdim=True).flatten(1),
            gray.amin((1, 2, 3), keepdim=True).flatten(1),
            horizontal.amax(2).flatten(1),
            vertical.amax(2).flatten(1),
            horizontal.std(2).flatten(1),
            vertical.std(2).flatten(1),
        ), dim=1)
        return torch.cat((horizontal_bins, vertical_bins, statistics), dim=1)


class PeakAwareImageEncoder(nn.Module):
    """ResNet-18 vision features fused with low-dimensional peak-profile features."""
    output_dim = 512

    def __init__(self, use_peak_features: bool = True, simclr_checkpoint: str | Path | None = None) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.vision = nn.Sequential(*list(backbone.children())[:-1])
        self.use_peak_features = use_peak_features
        self.peaks = RHEEDPeakFeatureExtractor()
        self.peak_projection = nn.Sequential(nn.Linear(self.peaks.output_dim, 128), nn.ReLU(), nn.Linear(128, 512))
        self.fusion = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.10))
        checkpoint = Path(simclr_checkpoint) if simclr_checkpoint else Path(__file__).resolve().parents[1] / "code" / "classifier2" / "simclr_encoder_checkpoint" / "simclr_resnet18_encoder.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"The required RHEED SimCLR checkpoint is missing: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("The RHEED SimCLR checkpoint is not a state dictionary.")
        cleaned = {key.replace("encoder.", ""): value for key, value in state.items() if not key.startswith("projector.")}
        target = self.vision.state_dict()
        compatible = {key: value for key, value in cleaned.items() if key in target and target[key].shape == value.shape}
        if not compatible:
            raise ValueError("No compatible RHEED SimCLR tensors were loaded into the ResNet-18 encoder.")
        self.vision.load_state_dict(compatible, strict=False)
        for parameter in self.vision.parameters():
            parameter.requires_grad = False
        self.vision.eval()
        self.encoder_provenance = {
            "name": "laboratory_rheed_simclr_resnet18", "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256(checkpoint), "loaded_tensor_count": len(compatible),
            "expected_tensor_count": len(target), "pairwise_preference_labels_used_in_pretraining": False,
            "fine_tuning_policy": "ResNet-18 SimCLR vision backbone frozen; peak projection, fusion, reward, and quality heads trainable.",
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        vision = self.vision(images).flatten(1)
        if not self.use_peak_features:
            return vision
        return self.fusion(torch.cat((vision, self.peak_projection(self.peaks(images))), dim=1))

    def train(self, mode: bool = True):
        """Keep frozen SimCLR batch-normalisation statistics fixed during head training."""
        super().train(mode)
        self.vision.eval()
        return self


class PeakAwareStaticRHEEDRewardModel(nn.Module):
    """Five Bradley--Terry reconstruction rewards plus a separate image-quality score."""
    def __init__(self, use_peak_features: bool = True, hidden_dim: int = 256, simclr_checkpoint: str | Path | None = None) -> None:
        super().__init__()
        self.encoder = PeakAwareImageEncoder(use_peak_features=use_peak_features, simclr_checkpoint=simclr_checkpoint)
        self.reward_head = nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU(), nn.Dropout(0.10), nn.Linear(hidden_dim, 5))
        self.quality_head = nn.Sequential(nn.Linear(512, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def reward_from_embeddings(self, embeddings: torch.Tensor, metadata=None) -> torch.Tensor:
        del metadata
        return self.reward_head(embeddings)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.reward_from_embeddings(self.encode(images))

    def quality_score(self, images: torch.Tensor) -> torch.Tensor:
        return self.quality_head(self.encode(images)).squeeze(1)

    @property
    def encoder_provenance(self) -> dict:
        return self.encoder.encoder_provenance

    def pairwise_probability(self, left: torch.Tensor, right: torch.Tensor, reconstruction_index: int) -> torch.Tensor:
        return torch.sigmoid(self(left)[:, reconstruction_index] - self(right)[:, reconstruction_index])

#!/usr/bin/env python3
"""Feature extractors for the classification baselines.  Every extractor maps image paths -> an (n, d) float array; results are cached
on disk per extractor (keyed by the path relative to the data root), so each image is encoded once per extractor.

Extractors that need downloaded weights (ImageNet / DINOv2 / CLIP) are registered but raise ``WeightsUnavailable`` if the weights cannot
be fetched; the runner records that as "unavailable" instead of silently substituting another model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

import baseline_protocol as proto

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "results" / "features"
SIMCLR_CHECKPOINT = HERE.parents[1] / "classifier2" / "simclr_encoder_checkpoint" / "simclr_resnet18_encoder.pth"
sys.path.insert(0, str(HERE.parents[2]))  # repository root, for paper_replicate


class WeightsUnavailable(RuntimeError):
    pass


def paper_transform() -> T.Compose:
    """Exactly ``pairwise_active_learning_pipeline.transform`` (grayscale -> 3 channels, 224, mean .5 / std .25)."""
    from pairwise_active_learning_pipeline import transform
    return transform()


def imagenet_transform() -> T.Compose:
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def encode(model: nn.Module, paths: list[Path], tf, batch: int = 16) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([tf(Image.open(p).convert("L")) for p in paths[i:i + batch]])
        out.append(model(x).flatten(1).float().numpy())
    return np.concatenate(out)


def resnet18_encoder(init: str, seed: int = 0) -> nn.Module:
    """512-d ResNet-18 encoder with the same layout the SimCLR checkpoint uses (children()[:-1])."""
    if init == "imagenet":
        try: backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as error: raise WeightsUnavailable(f"ImageNet ResNet-18 weights not downloadable: {error}") from error
    else:
        torch.manual_seed(seed); backbone = models.resnet18(weights=None)
    encoder = nn.Sequential(*list(backbone.children())[:-1])
    if init == "simclr":
        state = torch.load(SIMCLR_CHECKPOINT, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        cleaned = {k.replace("encoder.", ""): v for k, v in state.items() if not k.startswith("projector.")}
        missing, _ = encoder.load_state_dict(cleaned, strict=False)
        if missing: raise RuntimeError(f"SimCLR checkpoint left {len(missing)} encoder tensors random: {missing[:5]}")
    return encoder


def torchvision_imagenet(arch: str) -> nn.Module:
    """Penultimate-layer features of a torchvision ImageNet model (ResNet-50, ViT-B/16)."""
    try:
        if arch == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Identity()
        elif arch == "vit_b_16":
            m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1); m.heads = nn.Identity()
        else: raise ValueError(arch)
    except ValueError: raise
    except Exception as error: raise WeightsUnavailable(f"ImageNet {arch} weights not downloadable: {error}") from error
    return m


def raw_pixels(paths: list[Path], size: int = 64) -> np.ndarray:
    return np.stack([np.asarray(Image.open(p).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel() / 255. for p in paths])


def peak_profile(paths: list[Path]) -> np.ndarray:
    """The 24-d peak-aware intensity-profile features of the replicated peak-aware sequence paper (no learning)."""
    from paper_replicate.peak_aware_static_rheed_reward_model import RHEEDPeakFeatureExtractor
    tf = paper_transform(); extractor = RHEEDPeakFeatureExtractor()
    return np.concatenate([extractor(torch.stack([tf(Image.open(p).convert("L")) for p in paths[i:i + 16]])).numpy() for i in range(0, len(paths), 16)])


# name -> (kind used by prepare(), factory(paths) -> features, description)
EXTRACTORS = {
    "raw_pixels_64": ("tabular", raw_pixels, "64x64 grayscale pixels, z-scored with reference statistics"),
    "peak_profile_24": ("tabular", peak_profile, "24-d horizontal/vertical intensity-profile statistics (peak-aware paper features, untrained)"),
    **{f"random_resnet18_init{s}": ("deep", (lambda paths, s=s: encode(resnet18_encoder("random", s), paths, paper_transform())),
                                    f"ResNet-18, random weights (seed {s}), paper preprocessing") for s in range(3)},
    "simclr_resnet18": ("deep", lambda paths: encode(resnet18_encoder("simclr"), paths, paper_transform()), "laboratory RHEED SimCLR ResNet-18 (frozen)"),
    "imagenet_resnet18_paper_norm": ("deep", lambda paths: encode(resnet18_encoder("imagenet"), paths, paper_transform()),
                                     "ImageNet ResNet-18 with the paper's mean .5/std .25 preprocessing (what the earlier ImageNet runs used)"),
    "imagenet_resnet18": ("deep", lambda paths: encode(resnet18_encoder("imagenet"), paths, imagenet_transform()), "ImageNet ResNet-18, ImageNet preprocessing"),
    "imagenet_resnet50": ("deep", lambda paths: encode(torchvision_imagenet("resnet50"), paths, imagenet_transform()), "ImageNet ResNet-50 (torchvision V2 weights)"),
    "imagenet_vit_b_16": ("deep", lambda paths: encode(torchvision_imagenet("vit_b_16"), paths, imagenet_transform()), "ImageNet ViT-B/16 (torchvision weights)"),
}


def cached_features(name: str, paths: list[Path], data_root: Path = proto.DATA) -> np.ndarray:
    keys = [str(Path(p).resolve().relative_to(Path(data_root).resolve())) for p in paths]
    file = FEATURES / f"{name}.npz"
    store = {k: v for k, v in np.load(file).items()} if file.exists() else {}
    missing = [(k, p) for k, p in zip(keys, paths) if k not in store]
    if missing:
        torch.set_num_threads(2)
        features = EXTRACTORS[name][1]([p for _, p in missing])
        for (k, _), f in zip(missing, features): store[k] = f
        FEATURES.mkdir(parents=True, exist_ok=True); np.savez_compressed(file, **store)
    return np.stack([store[k] for k in keys])


def availability_note(name: str, error: Exception) -> dict:
    return {"extractor": name, "status": "unavailable", "reason": str(error)[:300]}

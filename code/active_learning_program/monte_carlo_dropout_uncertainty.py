"""MC-dropout epistemic uncertainty for cached Bradley--Terry embeddings."""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any

import torch
import torch.nn as nn


@contextmanager
def dropout_only_inference(model: nn.Module):
    """Enable only dropout while keeping the encoder and batch norm in eval mode."""
    states = {module: module.training for module in model.modules()}
    model.eval()
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train(True)
    try:
        yield
    finally:
        for module, was_training in states.items():
            module.train(was_training)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp(1e-7, 1 - 1e-7)
    return -(probabilities * probabilities.log() + (1 - probabilities) * (1 - probabilities).log())


def score_mc_dropout(
    candidates: list[dict[str, Any]],
    model: nn.Module,
    device: torch.device | str,
    embedding_cache: dict[str, torch.Tensor],
    mc_samples: int = 20,
    seed: int = 42,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], float]:
    """Return all three MC-dropout scores for every candidate.

    The frozen image encoder is never called here: the only stochastic work is
    repeated forwarding of cached [512]-dimensional features through reward_head.
    """
    if mc_samples < 2:
        raise ValueError("mc_samples must be at least 2 to estimate variance.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    model = model.to(device)
    scored: list[dict[str, Any]] = []
    with torch.no_grad(), dropout_only_inference(model):
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            left = torch.stack([embedding_cache[x["img1"]] for x in batch]).to(device)
            right = torch.stack([embedding_cache[x["img2"]] for x in batch]).to(device)
            differences = torch.stack([model.reward_head(left) - model.reward_head(right) for _ in range(mc_samples)])
            probabilities = torch.sigmoid(differences)
            probability_variance = probabilities.var(dim=0, unbiased=False).mean(dim=1)
            reward_variance = differences.var(dim=0, unbiased=False).mean(dim=1)
            mean_probability = probabilities.mean(dim=0)
            mutual_information = (_entropy(mean_probability) - _entropy(probabilities).mean(dim=0)).mean(dim=1)
            for item, pvar, mi, rvar in zip(batch, probability_variance.cpu().tolist(), mutual_information.cpu().tolist(), reward_variance.cpu().tolist()):
                scored.append({
                    **item,
                    "mc_probability_variance": float(pvar),
                    "mc_mutual_information": float(mi),
                    "mc_reward_variance": float(rvar),
                    "mc_samples": mc_samples,
                })
    return scored, time.perf_counter() - started


def select_mc_dropout(scored: list[dict[str, Any]], metric: str, budget: int) -> list[dict[str, Any]]:
    key = {
        "mc_dropout_probability_variance": "mc_probability_variance",
        "mc_dropout_mutual_information": "mc_mutual_information",
        "mc_dropout_reward_variance": "mc_reward_variance",
    }.get(metric)
    if key is None:
        raise ValueError(f"Unknown MC-dropout strategy: {metric}")
    return sorted(scored, key=lambda row: row[key], reverse=True)[:budget]

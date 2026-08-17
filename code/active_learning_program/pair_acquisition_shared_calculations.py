"""Selection and reporting primitives for controlled experiments."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def robust_unit(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] using 5th/95th percentile clipping."""
    if len(values) == 0: return values
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo: return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def pair_vector(item: dict[str, Any], cache: dict[str, torch.Tensor]) -> np.ndarray:
    return ((cache[item["img1"]].numpy() + cache[item["img2"]].numpy()) / 2).astype(np.float32)


def _minimum_distances(vectors: np.ndarray, covered: np.ndarray) -> np.ndarray:
    if len(covered) == 0: return np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
    return np.sqrt(((vectors[:, None, :] - covered[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def uncertainty_diversity_select(
    scored: list[dict[str, Any]], labeled_vectors: np.ndarray, budget: int, diversity_lambda: float
) -> list[dict[str, Any]]:
    """Greedy entropy + distance selection with audit fields on every choice."""
    if not scored: return []
    vectors = np.stack([x["pair_vector"] for x in scored])
    uncertainties = np.asarray([x["uncertainty"] for x in scored], dtype=float)
    normalized_uncertainty = robust_unit(uncertainties)
    available = list(range(len(scored))); covered = labeled_vectors.copy(); selected = []
    for rank in range(min(budget, len(scored))):
        subset = np.asarray(available)
        raw_distance = _minimum_distances(vectors[subset], covered)
        normalized_distance = robust_unit(raw_distance)
        combined = normalized_uncertainty[subset] + diversity_lambda * normalized_distance
        local = int(np.argmax(combined)); index = int(subset[local]); item = dict(scored[index])
        item.update({"raw_uncertainty": float(uncertainties[index]), "raw_diversity_distance": float(raw_distance[local]),
                     "normalized_uncertainty": float(normalized_uncertainty[index]), "normalized_diversity": float(normalized_distance[local]),
                     "diversity_lambda": float(diversity_lambda), "combined_score": float(combined[local]), "selection_rank": rank + 1})
        selected.append(item); covered = np.vstack([covered, vectors[index]]) if len(covered) else vectors[index:index + 1]
        available.remove(index)
    return selected


def core_set_select(candidates: list[dict[str, Any]], labeled_vectors: np.ndarray, budget: int) -> list[dict[str, Any]]:
    """Deterministic farthest-first k-center selection in pair embedding space."""
    if not candidates: return []
    vectors = np.stack([x["pair_vector"] for x in candidates]); available = list(range(len(candidates))); covered = labeled_vectors.copy(); selected = []
    for rank in range(min(budget, len(candidates))):
        subset = np.asarray(available); distances = _minimum_distances(vectors[subset], covered)
        local = int(np.argmax(distances)); index = int(subset[local]); item = dict(candidates[index])
        item.update({"core_set_min_distance": float(distances[local]), "selection_rank": rank + 1})
        selected.append(item); covered = np.vstack([covered, vectors[index]]) if len(covered) else vectors[index:index + 1]
        available.remove(index)
    return selected


def image_reuse_rate(selected: Iterable[dict[str, Any]]) -> float:
    paths = [path for item in selected for path in (item["img1"], item["img2"])]
    return 0.0 if not paths else 1 - len(set(paths)) / len(paths)


def cluster_entropy(selected: Iterable[dict[str, Any]]) -> float:
    labels = [x.get("cluster1") for x in selected if "cluster1" in x]
    if not labels: return float("nan")
    counts = np.asarray(list(Counter(labels).values()), dtype=float); p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def mean_pair_cosine_similarity(selected: list[dict[str, Any]]) -> float:
    vectors = np.asarray([x["pair_vector"] for x in selected], dtype=float)
    if len(vectors) < 2: return float("nan")
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    matrix = vectors @ vectors.T
    return float(matrix[np.triu_indices(len(vectors), k=1)].mean())


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows: grouped.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    output = []
    for key, values in grouped.items():
        summary = dict(zip(group_keys, key)); summary["n"] = len(values)
        for metric in metrics:
            data = np.asarray([x[metric] for x in values if x.get(metric) is not None], dtype=float)
            summary[f"{metric}_mean"] = float(data.mean()) if len(data) else float("nan")
            summary[f"{metric}_std"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        output.append(summary)
    return output

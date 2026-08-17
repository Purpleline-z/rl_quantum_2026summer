"""Leakage-safe feature-space NMF diagnostics for RHEED mixture surrogates.

This module deliberately reports a surrogate, not a physical mixture fraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score


@dataclass
class NMFDiagnostic:
    model: NMF
    component_class: dict[int, str]
    one_to_one: bool
    silhouette: float | None


def fit_reference_nmf(features: np.ndarray, labels: Iterable[str], n_components: int = 4, seed: int = 42) -> tuple[NMFDiagnostic, np.ndarray]:
    """Fit only on allowed ideal-reference embeddings; never pass test embeddings."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(list(labels), dtype=str)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Reference features and labels must have matching 2-D/1-D lengths.")
    if np.any(features < -1e-8):
        raise ValueError("Feature-space NMF requires nonnegative encoder activations; refusing to clip signed features.")
    if len(features) < n_components:
        raise ValueError("Need at least n_components reference images for NMF.")
    model = NMF(n_components=n_components, init="nndsvda", max_iter=1000, random_state=seed)
    weights = model.fit_transform(np.maximum(features, 0))
    mapping: dict[int, str] = {}
    for component in range(n_components):
        means = {label: float(weights[labels == label, component].mean()) for label in sorted(set(labels))}
        mapping[component] = max(means, key=means.get)
    assignments = weights.argmax(axis=1)
    silhouette = None
    if len(set(assignments)) > 1 and len(features) > len(set(assignments)):
        silhouette = float(silhouette_score(weights, labels))
    diagnostic = NMFDiagnostic(model=model, component_class=mapping,
                               one_to_one=len(set(mapping.values())) == n_components,
                               silhouette=silhouette)
    return diagnostic, weights


def score_nmf_features(diagnostic: NMFDiagnostic, features: np.ndarray) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=float)
    if np.any(features < -1e-8):
        raise ValueError("Feature-space NMF requires nonnegative encoder activations; refusing to clip signed features.")
    weights = diagnostic.model.transform(np.maximum(features, 0))
    normalized = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(normalized * np.log(np.clip(normalized, 1e-12, None))).sum(axis=1) / np.log(normalized.shape[1])
    reconstructed = weights @ diagnostic.model.components_
    residual = np.linalg.norm(features - reconstructed, axis=1) / np.clip(np.linalg.norm(features, axis=1), 1e-12, None)
    return {"weights": normalized, "mixture_entropy": entropy, "off_basis_residual": residual,
            "mixture_score": entropy}

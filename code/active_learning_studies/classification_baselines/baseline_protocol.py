#!/usr/bin/env python3
"""Paper-protocol split, classification heads and metrics for the image-classification baselines.

The split is NOT re-implemented: it is produced by the same ``Experiment.load_and_split`` call (same seed, same ideal-image
partition, same content-identity de-duplication, ``exclude_all_ideal_identities_from_pairwise=True``) that the Task 3c
experiments and ``compute_frozen_encoder_nearest_neighbour_baseline.py`` use.  Per seed that gives, for the four active types,
about 88 reference images (the labelled ideal images the Bradley-Terry model also sees as anchors), 28 outer-test images and
28 utility-validation images.  Baselines may fit on the references only; the outer test is touched once, for scoring;
the utility-validation images are the only data allowed for choosing a baseline's hyper-parameters.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
sys.path.insert(0, str(HERE.parents[1] / "active_learning_program"))

PAPER_SEEDS = (42, 79, 123, 202, 303)           # the five seeds of the Task 3c report
EXTRA_SEEDS = tuple(range(400, 425))            # more random splits of the SAME protocol (supplementary robustness only)
CLASSES = ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PCA_ABOVE, PCA_DIMS = 1024, 64


def load_split(seed: int, data_root: str | Path | None = None) -> dict:
    """Return {'references','test','utility'}: dict class -> list[Path], plus content identities for leak checks."""
    from pairwise_active_learning_pipeline import Config, Experiment
    scratch = tempfile.mkdtemp()  # keeps Experiment from rewriting results/active_learning_v1.8_seed*/manifests
    exp = Experiment(Config(seed=seed, device="cpu", data_root=str(data_root) if data_root else None,
                            exclude_all_ideal_identities_from_pairwise=True, manifest_dir=scratch))
    exp.load_and_split()
    split = {"references": exp.references, "test": exp.test_images, "utility": exp.utility_images}
    split["identity"] = {name: {str(p): exp._content_identity(p) for ps in table.values() for p in ps} for name, table in split.items() if name != "identity"}
    return split


def flatten(table: dict[str, list[Path]]) -> tuple[list[Path], np.ndarray]:
    paths = [p for c in CLASSES if c in table for p in table[c]]
    labels = np.array([c for c in CLASSES if c in table for _ in table[c]])
    return paths, labels


def assert_no_overlap(split: dict) -> None:
    """Fail loudly if any content identity occurs in two of references / test / utility."""
    ids = {name: set(split["identity"][name].values()) for name in ("references", "test", "utility")}
    for a, b in (("references", "test"), ("references", "utility"), ("test", "utility")):
        shared = ids[a] & ids[b]
        if shared: raise AssertionError(f"{len(shared)} byte-identical images occur in both {a} and {b}")


# ----------------------------------------------------------------------------- feature preprocessing
def prepare(kind: str, ref: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """'deep': L2-normalise (what the existing 1-NN baseline does).  'tabular': z-score with reference statistics, then L2."""
    arrays = [ref, *others]
    if kind == "tabular":
        mean, std = ref.mean(0), ref.std(0) + 1e-8
        arrays = [(a - mean) / std for a in arrays]
    return [a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12) for a in arrays]


# ----------------------------------------------------------------------------- heads (fit on references only)
def predict_knn(ref_x: np.ndarray, ref_y: np.ndarray, x: np.ndarray, k: int = 1) -> np.ndarray:
    """Cosine k-NN on L2-normalised features; majority vote, ties broken by summed similarity."""
    sim = x @ ref_x.T; out = []
    for row in sim:
        idx = np.argsort(-row, kind="stable")[:k]; votes: dict[str, list[float]] = {}
        for j in idx: votes.setdefault(ref_y[j], []).append(row[j])
        out.append(max(votes, key=lambda c: (len(votes[c]), sum(votes[c]))))
    return np.array(out)


def choose_c_by_reference_cv(ref_x: np.ndarray, ref_y: np.ndarray, grid=C_GRID, folds: int = 5, seed: int = 0) -> float:
    """Inner stratified CV on the REFERENCES only (never on test or utility images) to pick the logistic-regression C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    folds = max(2, min(folds, int(min(np.unique(ref_y, return_counts=True)[1]))))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {c: cross_val_score(LogisticRegression(C=c, max_iter=2000), ref_x, ref_y, cv=cv).mean() for c in grid}
    best = max(scores.values())
    return min(c for c, s in scores.items() if s == best)  # smallest C among ties = strongest regularisation


def predict_logreg(ref_x: np.ndarray, ref_y: np.ndarray, x: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    mean, std = ref_x.mean(0), ref_x.std(0) + 1e-8
    zr, zx = (ref_x - mean) / std, (x - mean) / std
    if zr.shape[1] > PCA_ABOVE:  # very wide inputs (raw pixels): PCA fitted on the references only, otherwise lbfgs is impractically slow
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(PCA_DIMS, len(zr) - 1), random_state=seed).fit(zr); zr, zx = pca.transform(zr), pca.transform(zx)
    c = choose_c_by_reference_cv(zr, ref_y, seed=seed)
    return LogisticRegression(C=c, max_iter=5000).fit(zr, ref_y).predict(zx)


HEADS = {"1nn": lambda rx, ry, x: predict_knn(rx, ry, x, 1), "5nn": lambda rx, ry, x: predict_knn(rx, ry, x, 5), "logreg": predict_logreg}


# ----------------------------------------------------------------------------- metrics
def score(truth: np.ndarray, pred: np.ndarray, classes=CLASSES) -> dict:
    row = {"n_test": len(truth), "n_correct": int((truth == pred).sum()), "accuracy": float((truth == pred).mean())}
    recalls, f1s = [], []
    for c in classes:
        tp = int(((truth == c) & (pred == c)).sum()); fp = int(((truth != c) & (pred == c)).sum()); fn = int(((truth == c) & (pred != c)).sum())
        recall = tp / (tp + fn) if tp + fn else float("nan"); precision = tp / (tp + fp) if tp + fp else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0); recalls.append(recall)
        row[f"recall[{c}]"] = recall
        if c == "HTR": row["HTR_precision"] = precision; row["HTR_recall"] = recall
    row["macro_f1"] = float(np.nanmean(f1s)); row["balanced_accuracy"] = float(np.nanmean(recalls))
    return row

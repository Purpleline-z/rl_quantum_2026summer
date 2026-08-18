#!/usr/bin/env python3
"""Create reproducible, coordinate-space diagnostics used by the academic report's Section 5."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "representation_exploration"
COORDINATES = SOURCE / "coordinates"
OUT = SOURCE / "section5_diagnostics"
K = 5
REPRESENTATIONS = ("rheed_simclr_resnet18", "imagenet_resnet18")
PROJECTIONS = ("pca", "tsne")


def leave_one_out_predictions(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """5-NN prediction with each ideal image excluded from its own neighborhood."""
    distances = euclidean_distances(x, x)
    np.fill_diagonal(distances, np.inf)
    predictions = []
    for row in distances:
        nearest = np.argsort(row)[: min(K, len(row) - 1)]
        votes = pd.Series(labels[nearest]).value_counts()
        predictions.append(votes.index[0])
    return np.asarray(predictions)


def analyze_coordinate_file(path: Path, representation: str, projection: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    ideals = frame[frame["is_labeled"]].copy()
    trajectories = frame[~frame["is_labeled"]].copy()
    labels = ideals["label"].to_numpy()
    points = ideals[["x", "y"]].to_numpy(dtype=float)
    predicted = leave_one_out_predictions(points, labels)
    classes = sorted(pd.unique(labels))
    confusion = confusion_matrix(labels, predicted, labels=classes)
    confusion_rows = []
    for truth, row in zip(classes, confusion):
        for predicted_label, count in zip(classes, row):
            confusion_rows.append({"representation": representation, "projection": projection, "true_class": truth,
                                   "predicted_class": predicted_label, "count": int(count)})

    distance = euclidean_distances(points, points)
    per_class = []
    for label in classes:
        mask = labels == label
        indices = np.flatnonzero(mask)
        same = distance[np.ix_(indices, indices)]
        within = same[np.triu_indices_from(same, k=1)]
        other = distance[np.ix_(indices, np.flatnonzero(~mask))]
        nearest_other = other.min(axis=1)
        recall = float((predicted[mask] == label).mean())
        support = int(mask.sum())
        median_within = float(np.median(within)) if len(within) else float("nan")
        median_nearest_other = float(np.median(nearest_other))
        per_class.append({"representation": representation, "projection": projection, "class": label, "support": support,
                          "knn_recall_at_5": recall, "knn_precision_at_5": float((labels[predicted == label] == label).mean()) if (predicted == label).any() else float("nan"),
                          "median_within_class_distance_2d": median_within,
                          "median_nearest_other_class_distance_2d": median_nearest_other,
                          "separation_ratio_2d": median_nearest_other / median_within if median_within else float("nan")})

    nearest_ideal = distance.copy(); np.fill_diagonal(nearest_ideal, np.inf)
    threshold = float(np.quantile(nearest_ideal.min(axis=1), .95))
    trajectory_points = trajectories[["x", "y"]].to_numpy(dtype=float)
    nearest_trajectory = euclidean_distances(trajectory_points, points).min(axis=1)
    coverage = {"representation": representation, "projection": projection, "n_labelled_ideal": int(len(ideals)),
                "n_unlabelled_trajectory": int(len(trajectories)), "near_threshold_2d": threshold,
                "trajectory_near_fraction": float((nearest_trajectory <= threshold).mean()),
                "trajectory_far_fraction": float((nearest_trajectory > threshold).mean()),
                "trajectory_median_nearest_ideal_distance_2d": float(np.median(nearest_trajectory)),
                "threshold_definition": "95th percentile of leave-one-out nearest-labelled-ideal distances in this 2D projection"}
    return pd.DataFrame(per_class), pd.DataFrame(confusion_rows), pd.DataFrame([coverage]), {"classes": classes, "matrix": confusion}


def write_figures(per_class: pd.DataFrame, confusion: pd.DataFrame, coverage: pd.DataFrame) -> None:
    pca = per_class[per_class.projection == "pca"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(pca["class"].unique()))
    classes = sorted(pca["class"].unique())
    width = .36
    for offset, representation in ((-width / 2, "rheed_simclr_resnet18"), (width / 2, "imagenet_resnet18")):
        values = pca[pca.representation == representation].set_index("class").loc[classes, "knn_recall_at_5"]
        ax.bar(positions + offset, values, width, label=representation.replace("rheed_", "").replace("_resnet18", ""))
    ax.set(xticks=positions, xticklabels=classes, ylim=(0, 1), ylabel="Leave-one-out 5-NN recall in PCA coordinates",
           title="Which ideal-image classes form locally label-consistent neighborhoods?")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(OUT / "per_class_knn_recall_pca.png", dpi=180); plt.close(fig)

    for representation in REPRESENTATIONS:
        subset = confusion[(confusion.representation == representation) & (confusion.projection == "pca")]
        classes = sorted(set(subset.true_class) | set(subset.predicted_class))
        matrix = subset.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=classes, columns=classes, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
               xlabel="5-NN predicted class", ylabel="True ideal class", title=f"{representation}: PCA-coordinate 5-NN confusion")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(len(classes)):
            for j in range(len(classes)): ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Ideal-image count"); fig.tight_layout()
        fig.savefig(OUT / f"{representation}_pca_knn_confusion.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{row.representation.replace('rheed_', '').replace('_resnet18', '')}\n{row.projection}" for row in coverage.itertuples()]
    ax.bar(labels, coverage.trajectory_near_fraction, label="near labelled-ideal neighbourhood")
    ax.bar(labels, coverage.trajectory_far_fraction, bottom=coverage.trajectory_near_fraction, label="far from labelled ideals")
    ax.set(ylim=(0, 1), ylabel="Fraction of unlabelled trajectory frames", title="Trajectory coverage in two-dimensional feature views")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "trajectory_neighborhood_coverage.png", dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_per_class, all_confusion, all_coverage, files = [], [], [], []
    for representation in REPRESENTATIONS:
        for projection in PROJECTIONS:
            path = COORDINATES / f"{representation}_{projection}.csv"
            per_class, confusion, coverage, _ = analyze_coordinate_file(path, representation, projection)
            all_per_class.append(per_class); all_confusion.append(confusion); all_coverage.append(coverage); files.append(str(path))
    per_class = pd.concat(all_per_class, ignore_index=True); confusion = pd.concat(all_confusion, ignore_index=True); coverage = pd.concat(all_coverage, ignore_index=True)
    per_class.to_csv(OUT / "per_class_knn_metrics.csv", index=False)
    confusion.to_csv(OUT / "knn_confusion_matrix.csv", index=False)
    per_class[["representation", "projection", "class", "median_within_class_distance_2d", "median_nearest_other_class_distance_2d", "separation_ratio_2d"]].to_csv(OUT / "class_separation_metrics.csv", index=False)
    coverage.to_csv(OUT / "trajectory_neighborhood_coverage.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"input_coordinate_files": files, "k": K, "distance_metric": "Euclidean distance in saved 2D coordinates", "near_threshold": "95th percentile of leave-one-out nearest-labelled-ideal distances per representation/projection", "scope": "descriptive diagnostics only; not used for model selection"}, indent=2), encoding="utf-8")
    write_figures(per_class, confusion, coverage)
    print(f"Wrote Section 5 representation diagnostics to {OUT}")


if __name__ == "__main__":
    main()

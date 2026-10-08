#!/usr/bin/env python3
"""Reference baseline for Task 3: frozen encoder + 1-nearest-neighbour on the reference images.

No pairwise labels and no training are used.  Each outer-test ideal image is assigned the class of
its most similar reference image (cosine similarity of frozen 512-d encoder features).  The splits
are the same identity-safe ideal-image partitions used by Task 3c for the same seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "active_learning_program"))
from pairwise_active_learning_pipeline import Config, Experiment, transform  # noqa: E402

DEFAULT_OUT = HERE / "results/simclr_three_seed_identity_safe_task3/frozen_encoder_nearest_neighbour_baseline.csv"


@torch.no_grad()
def embed(model, paths, tf):
    batches = [torch.stack([tf(Image.open(p).convert("L")) for p in paths[i:i + 16]]) for i in range(0, len(paths), 16)]
    features = torch.cat([model.encoder(batch) for batch in batches]).numpy()
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 79, 123, 202, 303])
    parser.add_argument("--encoder", choices=("simclr", "imagenet"), default="simclr")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows, tf = [], transform()
    for seed in args.seeds:
        exp = Experiment(Config(seed=seed, device="cpu", data_root=args.data_root, encoder_initialization=args.encoder,
                                exclude_all_ideal_identities_from_pairwise=True))
        exp.load_and_split()
        model = exp.make_model().eval()
        reference_features = np.concatenate([embed(model, paths, tf) for paths in exp.references.values()])
        reference_labels = np.array([name for name, paths in exp.references.items() for _ in paths])
        correct = total = 0
        for truth, paths in exp.test_images.items():
            predicted = reference_labels[(embed(model, paths, tf) @ reference_features.T).argmax(axis=1)]
            correct += int((predicted == truth).sum()); total += len(paths)
        rows.append({"seed": seed, "encoder": args.encoder, "outer_test_correct": correct, "outer_test_total": total,
                     "outer_test_accuracy": correct / total})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(args.output, index=False)
    print(f"mean {frame.outer_test_accuracy.mean():.3f} sd {frame.outer_test_accuracy.std():.3f}; wrote {args.output}")


if __name__ == "__main__":
    main()

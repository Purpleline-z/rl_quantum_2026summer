#!/usr/bin/env python3
"""Supervised fine-tuning baseline: ResNet-18 + linear 4-way head, cross-entropy on the REFERENCE images only (no pairwise labels).

Same split as the paper (``baseline_protocol.load_split``).  The number of epochs (1..MAX_EPOCHS) is chosen on the utility-validation images,
the only data the protocol allows for tuning; the outer test is evaluated once, at the selected epoch.  Initialisations: the laboratory
SimCLR checkpoint and random weights.  Writes metrics.csv / predictions.csv in the same format as run_baselines.py (head = 'finetune_ce').

  PYTHONPATH=../../active_learning_program python3 finetune_baseline.py --seeds paper --inits simclr,random --out results/finetune
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

import baseline_features as feats
import baseline_protocol as proto
from run_baselines import parse_seeds

HERE = Path(__file__).resolve().parent
MAX_EPOCHS, LR, WEIGHT_DECAY, BATCH = 30, 1e-4, 1e-4, 16


def select_epoch(utility_accuracy: list[float]) -> int:
    """1-based epoch with the best utility-validation accuracy; ties go to the earliest epoch."""
    return int(np.argmax(utility_accuracy)) + 1


def load_tensors(paths: list[Path]) -> torch.Tensor:
    tf = feats.paper_transform()
    return torch.stack([tf(Image.open(p).convert("L")) for p in paths])


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, classes: list[str]) -> np.ndarray:
    model.eval(); logits = torch.cat([model(x[i:i + 32]) for i in range(0, len(x), 32)])
    return np.array(classes)[logits.argmax(1).numpy()]


def finetune(init: str, seed: int, split: dict) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    classes = list(proto.CLASSES)
    ref_paths, ref_y = proto.flatten(split["references"]); util_paths, util_y = proto.flatten(split["utility"]); test_paths, test_y = proto.flatten(split["test"])
    x_ref, x_util, x_test = load_tensors(ref_paths), load_tensors(util_paths), load_tensors(test_paths)
    y_ref = torch.tensor([classes.index(c) for c in ref_y])
    model = nn.Sequential(feats.resnet18_encoder(init, seed), nn.Flatten(), nn.Linear(512, len(classes)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, utility_curve, best = None, [], -1.0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); order = torch.randperm(len(x_ref))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            if len(idx) < 2: continue  # BatchNorm needs more than one row
            optimizer.zero_grad(); nn.functional.cross_entropy(model(x_ref[idx]), y_ref[idx]).backward(); optimizer.step()
        utility_curve.append(float((predict(model, x_util, classes) == util_y).mean()))
        if utility_curve[-1] > best: best, best_state = utility_curve[-1], copy.deepcopy(model.state_dict())  # strict '>' keeps the earliest best epoch
    chosen = select_epoch(utility_curve); model.load_state_dict(best_state)
    pred = predict(model, x_test, classes)  # the one and only look at the outer test
    return {"pred": pred, "truth": test_y, "paths": test_paths, "epoch": chosen, "utility_accuracy": best, "n_reference": len(ref_y)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="paper"); parser.add_argument("--inits", default="simclr,random"); parser.add_argument("--out", type=Path, default=HERE / "results" / "finetune")
    args = parser.parse_args(); torch.set_num_threads(2); args.out.mkdir(parents=True, exist_ok=True)
    mfile, pfile = args.out / "metrics.csv", args.out / "predictions.csv"
    metrics = pd.read_csv(mfile) if mfile.exists() else pd.DataFrame(); predictions = pd.read_csv(pfile) if pfile.exists() else pd.DataFrame()
    done = set(zip(metrics.seed, metrics.extractor)) if len(metrics) else set()
    for seed in parse_seeds(args.seeds):
        split = proto.load_split(seed); proto.assert_no_overlap(split)
        for init in args.inits.split(","):
            name = f"finetune_{init}_resnet18"
            if (seed, name) in done: continue
            t = time.time(); out = finetune(init, seed, split)
            row = {"seed": seed, "extractor": name, "head": "finetune_ce", "n_reference": out["n_reference"], **proto.score(out["truth"], out["pred"]),
                   "selected_epoch": out["epoch"], "utility_accuracy": out["utility_accuracy"]}
            metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
            predictions = pd.concat([predictions, pd.DataFrame({"seed": seed, "extractor": name, "head": "finetune_ce", "image": [str(Path(p).relative_to(proto.DATA)) for p in out["paths"]],
                                                                "truth": out["truth"], "prediction": out["pred"]})], ignore_index=True)
            metrics.to_csv(mfile, index=False); predictions.to_csv(pfile, index=False)
            print(f"seed {seed} {name}: epoch {out['epoch']} utility {out['utility_accuracy']:.3f} -> test {row['accuracy']:.3f} macroF1 {row['macro_f1']:.3f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

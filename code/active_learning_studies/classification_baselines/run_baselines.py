#!/usr/bin/env python3
"""Run the frozen-feature classification baselines on the paper's exact per-seed split.

  python3 run_baselines.py --seeds paper|extra|42,79 --extractors simclr_resnet18,raw_pixels_64 --heads 1nn,5nn,logreg --out results/frozen

Writes <out>/metrics.csv (one row per seed x extractor x head), <out>/predictions.csv (one row per test image) and <out>/unavailable.json.
Rows already present in metrics.csv are skipped, so an interrupted run can be repeated unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_features as feats
import baseline_protocol as proto

HERE = Path(__file__).resolve().parent


def parse_seeds(text: str) -> tuple[int, ...]:
    if text == "paper": return proto.PAPER_SEEDS
    if text == "extra": return proto.EXTRA_SEEDS
    return tuple(int(s) for s in text.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="paper"); parser.add_argument("--extractors", default=",".join(feats.EXTRACTORS))
    parser.add_argument("--heads", default="1nn,5nn,logreg"); parser.add_argument("--out", type=Path, default=HERE / "results" / "frozen")
    args = parser.parse_args()
    seeds, names, heads = parse_seeds(args.seeds), args.extractors.split(","), args.heads.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_file, predictions_file, unavailable_file = args.out / "metrics.csv", args.out / "predictions.csv", args.out / "unavailable.json"
    metrics = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()
    predictions = pd.read_csv(predictions_file) if predictions_file.exists() else pd.DataFrame()
    unavailable = {u["extractor"]: u for u in json.loads(unavailable_file.read_text())} if unavailable_file.exists() else {}
    done = set(zip(metrics.seed, metrics.extractor, metrics["head"])) if len(metrics) else set()
    for seed in seeds:
        split = proto.load_split(seed); proto.assert_no_overlap(split)
        ref_paths, ref_y = proto.flatten(split["references"]); test_paths, test_y = proto.flatten(split["test"])
        print(f"seed {seed}: {len(ref_paths)} references, {len(test_paths)} test images", flush=True)
        for name in names:
            todo = [h for h in heads if (seed, name, h) not in done]
            if not todo or name in unavailable: continue
            try:
                ref_f, test_f = feats.cached_features(name, ref_paths), feats.cached_features(name, test_paths)
            except feats.WeightsUnavailable as error:
                unavailable[name] = feats.availability_note(name, error); print(f"  {name}: UNAVAILABLE ({error})", flush=True); continue
            kind = feats.EXTRACTORS[name][0]
            for head in todo:
                if head == "logreg": pred = proto.predict_logreg(ref_f, ref_y, test_f, seed=seed)  # standardises internally
                else: rx, tx = proto.prepare(kind, ref_f, test_f); pred = proto.HEADS[head](rx, ref_y, tx)
                row = {"seed": seed, "extractor": name, "head": head, "n_reference": len(ref_y), **proto.score(test_y, pred)}
                metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
                predictions = pd.concat([predictions, pd.DataFrame({"seed": seed, "extractor": name, "head": head, "image": [str(Path(p).relative_to(proto.DATA)) for p in test_paths],
                                                                    "truth": test_y, "prediction": pred})], ignore_index=True)
                print(f"  {name:32s} {head:7s} acc {row['accuracy']:.3f}  macroF1 {row['macro_f1']:.3f}  HTR recall {row['HTR_recall']:.2f}", flush=True)
            metrics.to_csv(metrics_file, index=False); predictions.to_csv(predictions_file, index=False)
    unavailable_file.write_text(json.dumps(list(unavailable.values()), indent=2))
    metrics.to_csv(metrics_file, index=False); predictions.to_csv(predictions_file, index=False)


if __name__ == "__main__":
    main()

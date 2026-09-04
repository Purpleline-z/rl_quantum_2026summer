"""Evaluate a fixed reward model only on the sealed image-disjoint test split."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from PIL import Image

from .peak_aware_static_rheed_reward_model import PeakAwareStaticRHEEDRewardModel, RECONSTRUCTION_TYPES
from .pairwise_and_absolute_label_dataset import load_pairwise_rows
from .train_peak_aware_reconstruction_reward_model import TRANSFORM, TYPE_INDEX


def evaluate_sealed_test_set(data_root, split, weights_path, output_directory, use_peak_features=True, device="cpu"):
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    test_images = set(split["images"]["test"])
    rows = [r for r in load_pairwise_rows(data_root) if r["left"] in test_images and r["right"] in test_images]
    model = PeakAwareStaticRHEEDRewardModel(use_peak_features=use_peak_features).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)); model.eval()
    correct = denominator = 0; details = []
    with torch.no_grad():
        for row in rows:
            left = TRANSFORM(Image.open(row["left"]).convert("RGB")).unsqueeze(0).to(device)
            right = TRANSFORM(Image.open(row["right"]).convert("RGB")).unsqueeze(0).to(device)
            index = TYPE_INDEX.get(row["reconstruction_type"], 0)
            probability = float(model.pairwise_probability(left, right, index).cpu())
            predicted = "1" if probability > .5 else "2"
            if row["winner"] in {"1", "2"}:
                denominator += 1; correct += predicted == row["winner"]
            entropy = -(probability * __import__("math").log(max(probability, 1e-8)) + (1-probability) * __import__("math").log(max(1-probability, 1e-8)))
            details.append({"pair_id": row["pair_id"], "winner": row["winner"], "predicted_winner": predicted, "probability_left_wins": probability, "entropy": entropy})
    result = {"status": "completed", "sealed_test_pairs": len(rows), "decisive_test_pairs": denominator,
              "pairwise_winner_accuracy": correct / denominator if denominator else None,
              "test_image_policy": split["test_image_policy"], "results": details}
    (output / "sealed_test_pairwise_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "completed_task_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

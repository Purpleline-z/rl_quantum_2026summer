"""Sealed session-test evaluation and validation-only calibration for reward models."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
import torch
from PIL import Image

from .peak_aware_static_rheed_reward_model import PeakAwareStaticRHEEDRewardModel
from .pairwise_and_absolute_label_dataset import load_pairwise_rows
from .train_peak_aware_reconstruction_reward_model import TRANSFORM, TYPE_INDEX


def _macro_f1(labels, predictions):
    values = []
    for label in ("1", "2", "tie", "not_apply"):
        true_positive = sum(a == label and b == label for a, b in zip(labels, predictions))
        false_positive = sum(a != label and b == label for a, b in zip(labels, predictions))
        false_negative = sum(a == label and b != label for a, b in zip(labels, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.)
    return sum(values) / len(values)


def fit_validation_calibration(validation_records):
    """Choose temperature and tie/not-applicable thresholds without test data."""
    decisive = [row for row in validation_records if row["winner"] in {"1", "2"}]
    temperatures = [0.5, 0.75, 1., 1.5, 2., 3.]
    temperature = min(temperatures, key=lambda value: sum(
        (1. / (1. + math.exp(-row["margin"] / value)) - (1. if row["winner"] == "1" else 0.)) ** 2
        for row in decisive) / max(1, len(decisive)))
    strengths = [row["reward_strength"] for row in validation_records] or [0.]
    magnitudes = [abs(row["margin"]) for row in validation_records] or [0.]
    best = None
    for not_apply_strength in sorted(set([min(strengths) - 1e-6, *strengths, max(strengths) + 1e-6])):
        for tie_margin in sorted(set([0., *magnitudes, max(magnitudes) + 1e-6])):
            predictions = ["not_apply" if row["reward_strength"] < not_apply_strength else "tie" if abs(row["margin"]) <= tie_margin else ("1" if row["margin"] > 0 else "2") for row in validation_records]
            candidate = (_macro_f1([row["winner"] for row in validation_records], predictions), not_apply_strength, tie_margin)
            if best is None or candidate[0] > best[0]: best = candidate
    return {"temperature": temperature, "not_apply_reward_strength_threshold": best[1], "tie_reward_margin_threshold": best[2], "validation_macro_f1": best[0], "fitted_on": "validation_only"}


def _label_for_record(margin, strength, calibration):
    if strength < calibration["not_apply_reward_strength_threshold"]: return "not_apply"
    if abs(margin) <= calibration["tie_reward_margin_threshold"]: return "tie"
    return "1" if margin > 0 else "2"


def _metrics(records):
    labels, predictions = [r["winner"] for r in records], [r["predicted_label"] for r in records]
    decisive = [r for r in records if r["winner"] in {"1", "2"}]
    per_type = {}
    for reconstruction_type in sorted({r["reconstruction_type"] for r in records}):
        subset = [r for r in records if r["reconstruction_type"] == reconstruction_type]
        subset_decisive = [r for r in subset if r["winner"] in {"1", "2"}]
        per_type[reconstruction_type] = {"pair_count": len(subset), "decisive_pair_count": len(subset_decisive), "decisive_accuracy": sum(r["winner"] == r["predicted_label"] for r in subset_decisive) / len(subset_decisive) if subset_decisive else None, "label_counts": dict(Counter(r["winner"] for r in subset))}
    probabilities = [r["probability_left_wins"] for r in decisive]
    targets = [1. if r["winner"] == "1" else 0. for r in decisive]
    return {"all_label_accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels) if labels else None, "macro_f1": _macro_f1(labels, predictions) if labels else None, "decisive_pairwise_accuracy": sum(r["winner"] == r["predicted_label"] for r in decisive) / len(decisive) if decisive else None, "decisive_pair_count": len(decisive), "pair_count": len(records), "label_counts": dict(Counter(labels)), "per_reconstruction_type": per_type, "calibration_brier_score_on_decisive_pairs": sum((p - t) ** 2 for p, t in zip(probabilities, targets)) / len(targets) if targets else None}


def evaluate_sealed_test_set(data_root, split, weights_path, output_directory, use_peak_features=True, device="cpu", calibration=None, include_embeddings=False):
    """Read held-out session images only after architecture and calibration are fixed."""
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    test_images = set(split["images"]["test"])
    all_rows = load_pairwise_rows(data_root)
    role_pair_ids = split.get("pair_ids_by_role")
    rows = ([row for row in all_rows if row["pair_id"] in set(role_pair_ids["test"])] if role_pair_ids
            else [row for row in all_rows if row["left"] in test_images and row["right"] in test_images])
    model = PeakAwareStaticRHEEDRewardModel(use_peak_features=use_peak_features).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)); model.eval()
    calibration = calibration or {"temperature": 1., "not_apply_reward_strength_threshold": float("-inf"), "tie_reward_margin_threshold": 0.}
    details = []
    with torch.no_grad():
        for row in rows:
            left = TRANSFORM(Image.open(row["left"]).convert("RGB")).unsqueeze(0).to(device)
            right = TRANSFORM(Image.open(row["right"]).convert("RGB")).unsqueeze(0).to(device)
            index = TYPE_INDEX[row["reconstruction_type"]]
            left_rewards, right_rewards = model(left), model(right)
            margin = float((left_rewards[0, index] - right_rewards[0, index]).cpu())
            strength = float(torch.maximum(left_rewards[0, index], right_rewards[0, index]).cpu())
            probability = 1. / (1. + math.exp(-margin / calibration["temperature"]))
            record = {"pair_id": row["pair_id"], "reconstruction_type": row["reconstruction_type"], "winner": row["winner"], "predicted_label": _label_for_record(margin, strength, calibration), "probability_left_wins": probability, "reward_margin": margin, "reward_strength": strength, "entropy": -(probability * math.log(max(probability, 1e-8)) + (1 - probability) * math.log(max(1 - probability, 1e-8)))}
            if include_embeddings: record["pair_embedding"] = torch.cat((model.encode(left).squeeze(0), model.encode(right).squeeze(0))).cpu().tolist()
            details.append(record)
    result = {"status": "completed", "sealed_test_pairs": len(rows), "held_out_test_session": split.get("held_out_test_session"), "test_image_count": len(test_images), "test_image_manifest": sorted(test_images), "test_image_policy": split["test_image_policy"], "encoder_provenance": model.encoder_provenance, "calibration": calibration, "metrics": _metrics(details), "results": details}
    (output / "sealed_session_held_out_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "completed_task_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

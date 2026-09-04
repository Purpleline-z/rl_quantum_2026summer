"""Export compact reward, embedding, margin and entropy records for existing selectors."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from PIL import Image

from .peak_aware_static_rheed_reward_model import PeakAwareStaticRHEEDRewardModel
from .train_peak_aware_reconstruction_reward_model import TRANSFORM
from .pairwise_and_absolute_label_dataset import load_pairwise_rows


def export_pair_selection_features(data_root, weights_path, output_directory, use_peak_features=True, device="cpu"):
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    model = PeakAwareStaticRHEEDRewardModel(use_peak_features=use_peak_features).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)); model.eval()
    cache = {}
    def encoded(path):
        if path not in cache:
            image = TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad(): cache[path] = (model.encode(image).squeeze(0).cpu(), model(image).squeeze(0).cpu(), float(model.quality_score(image).cpu()))
        return cache[path]
    records = []
    with torch.no_grad():
        for row in load_pairwise_rows(data_root):
            left, right = encoded(row["left"]), encoded(row["right"])
            probabilities = torch.sigmoid(left[1] - right[1])
            entropy = (-(probabilities.clamp(1e-7, 1-1e-7) * probabilities.clamp(1e-7, 1-1e-7).log() + (1-probabilities).clamp(1e-7, 1-1e-7) * (1-probabilities).clamp(1e-7, 1-1e-7).log())).mean()
            records.append({"pair_id": row["pair_id"], "img1": row["left"], "img2": row["right"], "pair_embedding": torch.cat((left[0], right[0])).tolist(), "reward_margin": (left[1]-right[1]).tolist(), "uncertainty": float(entropy), "quality_scores": [left[2], right[2]]})
    result = {"status": "completed", "record_count": len(records), "records": records}
    (output / "active_learning_pair_selection_features.json").write_text(json.dumps(result), encoding="utf-8")
    (output / "completed_task_result.json").write_text(json.dumps({"status": "completed", "record_count": len(records)}, indent=2), encoding="utf-8")
    return result

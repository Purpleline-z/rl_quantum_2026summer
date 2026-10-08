"""The label-free wrapper must remove the candidate type and leave rankings of the active heads unchanged by the silenced Twinned head."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"))
from frozen_encoder_reward_head import label_free_inputs  # noqa: E402


class _Model:
    def __init__(self):
        torch.manual_seed(0); self.reward_head = nn.Sequential(nn.Linear(512, 16), nn.ReLU(), nn.Dropout(.2), nn.Linear(16, 5))
    def eval(self): self.reward_head.eval(); return self


def test_type_removed_and_twinned_head_silenced():
    model = _Model(); rows = [{"pair_id": "a", "img1": "x", "img2": "y", "type_idx": 3}]
    stripped, silenced = label_free_inputs(rows, model)
    assert "type_idx" not in stripped[0] and "type_idx" in rows[0]
    a, b = torch.randn(6, 512), torch.randn(6, 512)
    gap = silenced.reward_head.eval()(a) - silenced.reward_head.eval()(b)
    assert torch.allclose(gap[:, 1], torch.zeros(6)) and not torch.allclose(gap[:, 0], torch.zeros(6))
    assert not torch.equal(silenced.reward_head[3].weight, model.reward_head[3].weight)  # a copy, original untouched
    assert model.reward_head[3].weight[1].abs().sum() > 0

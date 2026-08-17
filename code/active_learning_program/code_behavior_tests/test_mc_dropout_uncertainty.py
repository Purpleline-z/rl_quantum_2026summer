"""Focused tests for the cached-feature MC-dropout implementation."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from monte_carlo_dropout_uncertainty import dropout_only_inference, score_mc_dropout, select_mc_dropout


class DummyBT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.BatchNorm1d(4))
        self.reward_head = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.5), nn.Linear(8, 5))


class McDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7); self.model = DummyBT()
        self.candidates = [{"pair_id": "a", "img1": "a1", "img2": "a2"}, {"pair_id": "b", "img1": "b1", "img2": "b2"}]
        self.cache = {key: torch.randn(4) for key in ("a1", "a2", "b1", "b2")}

    def test_dropout_only_keeps_batch_norm_in_eval(self):
        self.model.train()
        with dropout_only_inference(self.model):
            self.assertFalse(self.model.encoder[0].training)
            self.assertTrue(self.model.reward_head[2].training)
        self.assertTrue(self.model.encoder[0].training)

    def test_scores_are_seeded_finite_and_stochastic(self):
        one, _ = score_mc_dropout(self.candidates, self.model, "cpu", self.cache, mc_samples=12, seed=11)
        two, _ = score_mc_dropout(self.candidates, self.model, "cpu", self.cache, mc_samples=12, seed=11)
        for left, right in zip(one, two):
            for key in ("mc_probability_variance", "mc_mutual_information", "mc_reward_variance"):
                self.assertTrue(torch.isfinite(torch.tensor(left[key])))
                self.assertEqual(left[key], right[key])
            self.assertGreater(left["mc_reward_variance"], 0.0)

    def test_each_strategy_selects_its_highest_score(self):
        scored, _ = score_mc_dropout(self.candidates, self.model, "cpu", self.cache, mc_samples=12, seed=11)
        mapping = {"mc_dropout_probability_variance": "mc_probability_variance", "mc_dropout_mutual_information": "mc_mutual_information", "mc_dropout_reward_variance": "mc_reward_variance"}
        for strategy, key in mapping.items():
            selected = select_mc_dropout(scored, strategy, 1)
            self.assertEqual(selected[0][key], max(x[key] for x in scored))


if __name__ == "__main__":
    unittest.main()

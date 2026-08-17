from __future__ import annotations
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from pair_acquisition_shared_calculations import aggregate_rows, core_set_select, robust_unit, uncertainty_diversity_select
from pair_acquisition_methods import cluster_quota_uncertainty_sampling, cluster_margin_pairwise_sampling
from resumable_model_training import train_with_epoch_checkpoints
from run_initial_fixed_protocol_comparison import checkpointing_enabled


class FixedProtocolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"pair_id": str(i), "pair_vector": [float(i), 0.], "uncertainty": float(i), "img1": str(i), "img2": str(i)} for i in range(4)]

    def test_core_set_farthest_first(self):
        chosen = core_set_select(self.items, np.array([[0., 0.]]), 2)
        self.assertEqual([x["pair_id"] for x in chosen], ["3", "1"])

    def test_uncertainty_diversity_logs_and_unique(self):
        chosen = uncertainty_diversity_select(self.items, np.array([[0., 0.]]), 3, .5)
        self.assertEqual(len({x["pair_id"] for x in chosen}), 3)
        self.assertEqual([x["selection_rank"] for x in chosen], [1, 2, 3])
        self.assertTrue(all("combined_score" in x for x in chosen))

    def test_robust_normalization(self):
        values = robust_unit(np.array([1., 2., 3., 4., 5.]))
        self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_aggregate_mean_and_sample_std(self):
        summary = aggregate_rows([{"epoch": 1, "strategy": "x", "accuracy": .2}, {"epoch": 1, "strategy": "x", "accuracy": .4}], ["epoch", "strategy"], ["accuracy"])[0]
        self.assertAlmostEqual(summary["accuracy_mean"], .3)
        self.assertAlmostEqual(summary["accuracy_std"], np.sqrt(.02))

    def test_interaction_gap_definition(self):
        batch_utility, individual = .10, [.03, -.01, .02]
        self.assertAlmostEqual(batch_utility - sum(individual), .06)

    def test_cluster_quota_uses_each_cluster_before_fill(self):
        candidates = [{"img1": "a", "img2": "b", "cluster1": 0}, {"img1": "c", "img2": "d", "cluster1": 1}, {"img1": "e", "img2": "f", "cluster1": 1}]
        scored = [{**candidates[0], "uncertainty": .9}, {**candidates[1], "uncertainty": .8}, {**candidates[2], "uncertainty": .1}]
        with patch("pair_acquisition_methods.score_uncertainty", return_value=scored):
            selected = cluster_quota_uncertainty_sampling(candidates, None, 2)
        self.assertEqual({x["cluster1"] for x in selected}, {0, 1})

    def test_cluster_margin_starts_with_smallest_prefiltered_cluster(self):
        candidates = [{"pair_id": "a", "img1": "a1", "img2": "a2", "cluster1": 0},
                      {"pair_id": "b", "img1": "b1", "img2": "b2", "cluster1": 0},
                      {"pair_id": "c", "img1": "c1", "img2": "c2", "cluster1": 1}]
        class Model:
            metadata_dim = 0
            reward_head = staticmethod(lambda x: x)
            def to(self, device): return self
            def eval(self): return self
        cache = {"a1": torch.tensor([0.]), "a2": torch.tensor([1.]),
                 "b1": torch.tensor([0.]), "b2": torch.tensor([.4]),
                 "c1": torch.tensor([0.]), "c2": torch.tensor([.1])}
        with patch("pair_acquisition_methods.score_uncertainty", return_value=[{**x, "uncertainty": .5} for x in candidates]):
            selected = cluster_margin_pairwise_sampling(candidates, Model(), 2, embedding_cache=cache)
        self.assertEqual(selected[0]["pair_id"], "c")
        self.assertEqual([x["selection_rank"] for x in selected], [1, 2])
        self.assertTrue(all("cluster_margin" in candidate and "prefilter_member" in candidate for candidate in candidates))
        self.assertTrue(all("cluster_size" in candidate and "prefilter_cluster_size" in candidate for candidate in candidates))

    def test_fifty_minute_policy_disables_checkpoint_writes(self):
        class Exp:
            device = torch.device("cpu")
            class cfg:
                lr = 1e-4
                weight_decay = 1e-4
                train_batch_size = 1
                epochs = 0
            def rows_for(self, ids): return pd.DataFrame({"unused": [1]})
            def make_model(self): return torch.nn.Linear(1, 1)
            def pairwise_accuracy(self, model, rows): return 1.0
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, patch("resumable_model_training.atomic_torch_save") as save:
            checkpoint = Path(directory) / "short_job.pth"
            _, _, paused = train_with_epoch_checkpoints(Exp(), ["pair"], checkpoint, "phase", None, checkpoint_enabled=checkpointing_enabled(50))
            self.assertFalse(checkpointing_enabled(50))
            self.assertFalse(paused)
            save.assert_not_called()
            self.assertFalse(checkpoint.exists())

    def test_one_hundred_twenty_one_minute_policy_enables_checkpointing(self):
        self.assertTrue(checkpointing_enabled(121))
        class PairDataset(torch.utils.data.Dataset):
            def __init__(self, rows): pass
            def __len__(self): return 1
            def __getitem__(self, index):
                return torch.zeros(3, 1, 1), torch.ones(3, 1, 1), 0, "1", 1.0
        class Exp:
            device = torch.device("cpu")
            references = {}
            class cfg:
                lr = 1e-4
                weight_decay = 1e-4
                train_batch_size = 1
                epochs = 1
            def rows_for(self, ids): return pd.DataFrame({"unused": [1]})
            def make_model(self): return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3, 5))
            def pairwise_accuracy(self, model, rows): return 1.0
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, patch("pairwise_active_learning_pipeline.PairRows", PairDataset), patch("resumable_model_training.atomic_torch_save") as save:
            train_with_epoch_checkpoints(Exp(), ["pair"], Path(directory) / "long_job.pth", "phase", None, checkpoint_enabled=checkpointing_enabled(121))
            save.assert_called_once()


if __name__ == "__main__": unittest.main()

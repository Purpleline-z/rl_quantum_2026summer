from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from resumable_model_training import train_with_epoch_checkpoints


HERE = Path(__file__).resolve()
RUNNER = HERE.parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint" / "run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py"
CONFIG = RUNNER.with_name("fixed_3_epoch_vs_30_epoch_single_shot_settings.json")
spec = importlib.util.spec_from_file_location("fixed_epoch_runner", RUNNER)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class FixedEpochSingleShotBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = runner.read_json(CONFIG)

    def test_registered_grid_has_sixty_unique_cells(self):
        tasks = runner.all_tasks(self.protocol)
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len({runner.task_key(task) for task in tasks}), 60)
        self.assertEqual({task["strategy"] for task in tasks}, {"random", "uncertainty"})
        self.assertEqual({task["epochs"] for task in tasks}, {3, 30})

    def test_two_worker_assignment_is_complete_and_disjoint(self):
        left = {runner.task_key(task) for task in runner.assigned_tasks(self.protocol, 0, 2)}
        right = {runner.task_key(task) for task in runner.assigned_tasks(self.protocol, 1, 2)}
        self.assertEqual(len(left), 30)
        self.assertEqual(len(right), 30)
        self.assertFalse(left & right)
        self.assertEqual(left | right, {runner.task_key(task) for task in runner.all_tasks(self.protocol)})

    def test_cell_name_is_human_readable(self):
        task = runner.all_tasks(self.protocol)[0]
        self.assertEqual(runner.cell_filename(task), "seed_42_budget_10_strategy_random_fixed_3_epochs.json")

    def test_strict_audit_rejects_any_protected_identity_overlap(self):
        audit = {field: 0 for field in runner.ZERO_AUDIT_FIELDS}
        audit["candidate_labels_hidden_from_selector"] = True
        runner.assert_strict_audit(audit, self.protocol, list(range(10)), list(range(100)))
        audit["pairwise_image_identity_overlap_utility_validation"] = 1
        with self.assertRaises(RuntimeError):
            runner.assert_strict_audit(audit, self.protocol, list(range(10)), list(range(100)))

    def test_outer_test_is_only_recorded_at_its_registered_epoch(self):
        class OnePair(torch.utils.data.Dataset):
            def __init__(self, rows):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return torch.zeros(3, 1, 1), torch.ones(3, 1, 1), 0, "1", 1.0

        class Exp:
            device = torch.device("cpu")
            references = {}
            evaluated: list[str] = []

            class cfg:
                lr = 1e-4
                weight_decay = 1e-4
                train_batch_size = 1
                epochs = 1

            def rows_for(self, ids):
                return pd.DataFrame({"unused": [1]})

            def make_model(self):
                return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3, 5))

            def pairwise_accuracy(self, model, rows):
                return 1.0

            def evaluate(self, model, split):
                self.evaluated.append(split)
                return {"test_accuracy": .75, "test_correct": 3, "test_total": 4, "by_class": {}}

        experiment = Exp()
        with TemporaryDirectory() as directory, patch("pairwise_active_learning_pipeline.PairRows", OnePair):
            _, metrics, paused = train_with_epoch_checkpoints(
                experiment, ["pair"], Path(directory) / "checkpoint.pth", "registered-endpoint", None,
                checkpoint_enabled=False, evaluation_splits_by_epoch={1: ("outer_test",)},
                return_best_validation_model=False,
            )
        self.assertFalse(paused)
        self.assertEqual(experiment.evaluated, ["outer_test"])
        observed = metrics["epoch_metrics"][0]["pre_registered_evaluations"]["outer_test"]
        self.assertEqual(observed["accuracy"], .75)
        self.assertEqual(observed["total"], 4)


if __name__ == "__main__":
    unittest.main()

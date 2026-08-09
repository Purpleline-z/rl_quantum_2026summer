from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from active_learning_pipeline import Config, Experiment, canonical_type
from dataset_protocol import dataset_spec, locate_input


class DatasetProtocolTests(unittest.TestCase):
    def test_version_contract_does_not_fallback(self):
        self.assertEqual(dataset_spec("v5.7").pairwise_name, "Quantum Label Data - Pairwise_Comparisonv5.7.csv")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Quantum Label Data - Pairwise_Comparisonv1.8.csv").write_text("x", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                locate_input(root, root, dataset_spec("v5.7").pairwise_name)

    def test_root13_normalisation_accepts_unicode_and_legacy_text(self):
        self.assertEqual(canonical_type("(√13 x √13)"), canonical_type("RT13"))

    def test_candidate_view_has_no_preference_label_and_splits_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "original data"; data.mkdir()
            folders = {"(1 x 1)": "STO_ideal_1x1", "c(6 x 2)": "STO_ideal_c6x2", "HTR": "STO_ideal_HTR", "(√13 x √13)": "STO_ideal_RT13"}
            rows = []
            for index, (label, folder) in enumerate(folders.items()):
                target = data / folder; target.mkdir()
                for image_index in range(3): Image.new("L", (8, 8), 20 + image_index).save(target / f"{index}_{image_index}.png")
                a, b = f"{folder}/{index}_0.png", f"{folder}/{index}_1.png"
                rows.append({"Image1_Path": a, "Image2_Path": b, "Reconstruction_Type": label, "Winner": "1"})
                rows.append({"Image1_Path": f"{folder}/{index}_1.png", "Image2_Path": f"{folder}/{index}_2.png", "Reconstruction_Type": label, "Winner": "2"})
            pd.DataFrame(rows).to_csv(data / "Quantum Label Data - Pairwise_Comparisonv1.8.csv", index=False)
            pd.DataFrame(columns=["Reconstruction", "Image_Path"]).to_csv(data / "Quantum Label Data - Absolute_Scoringv1.8 (1).csv", index=False)
            exp = Experiment(Config(data_root=str(root), initial_pairs=4, candidate_pairs=4, dataset_version="v1.8"))
            initial, pool = exp.load_and_split()
            self.assertFalse(set(initial) & set(pool))
            self.assertTrue(exp.candidate_metadata)
            self.assertTrue(all("Winner" not in item and "canonical_type" not in item for item in exp.candidate_metadata.values()))
            test = {p for values in exp.test_images.values() for p in values}
            utility = {p for values in exp.utility_images.values() for p in values}
            references = {p for values in exp.references.values() for p in values}
            self.assertFalse(test & utility); self.assertFalse(test & references); self.assertFalse(utility & references)

    def test_controlled_utility_never_queries_outer_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Experiment(Config(data_root=tmp))
            exp.utility_cache_path = Path(tmp) / "utility_cache.json"
            exp.utility_cache = {}
            exp.train = Mock(return_value=(object(), {}))
            exp.evaluate = Mock(side_effect=[
                {"test_accuracy": .4, "by_class": {"HTR": {"accuracy": .4}}},
                {"test_accuracy": .6, "by_class": {"HTR": {"accuracy": .6}}},
            ])
            exp.controlled_utility(["known"], "candidate")
            self.assertEqual([call.kwargs["split"] for call in exp.evaluate.call_args_list], ["utility_validation", "utility_validation"])


if __name__ == "__main__":
    unittest.main()

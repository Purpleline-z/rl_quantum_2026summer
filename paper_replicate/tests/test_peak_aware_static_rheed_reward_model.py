import unittest
from pathlib import Path
import tempfile
import torch

from paper_replicate.peak_aware_static_rheed_reward_model import PeakAwareStaticRHEEDRewardModel, RHEEDPeakFeatureExtractor
from paper_replicate.evaluate_image_disjoint_reconstruction_reward_model import fit_validation_calibration
from paper_replicate.pairwise_and_absolute_label_dataset import (create_fixed_five_fold_unseen_image_splits,
    create_session_held_out_split, session_held_out_audit)

class PeakAwareModelTests(unittest.TestCase):
    def test_peak_features_are_compact_and_finite(self):
        values = RHEEDPeakFeatureExtractor()(torch.randn(2, 3, 64, 80))
        self.assertEqual(tuple(values.shape), (2, 24))
        self.assertTrue(torch.isfinite(values).all())

    def test_model_has_selector_compatible_outputs(self):
        model = PeakAwareStaticRHEEDRewardModel(); images = torch.randn(2, 3, 64, 64)
        embeddings = model.encode(images); rewards = model.reward_from_embeddings(embeddings)
        self.assertEqual(tuple(embeddings.shape), (2, 512))
        self.assertEqual(tuple(rewards.shape), (2, 5))
        self.assertEqual(tuple(model.quality_score(images).shape), (2,))
        self.assertEqual(tuple(model.pairwise_probability(images[:1], images[1:], 0).shape), (1,))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.vision.parameters()))
        self.assertEqual(model.encoder_provenance["loaded_tensor_count"], 120)

    def test_session_held_out_split_seals_entire_session(self):
        base = Path("/temporary/data/original data/Trajectories")
        rows = [
            {"pair_id": "a", "left": str(base / "2022-01-01" / "a.bmp"), "right": str(base / "2022-01-01" / "b.bmp"), "winner": "1", "reconstruction_type": "HTR"},
            {"pair_id": "b", "left": str(base / "2022-01-02" / "c.bmp"), "right": str(base / "2022-01-02" / "d.bmp"), "winner": "2", "reconstruction_type": "HTR"},
        ]
        self.assertTrue(session_held_out_audit(rows, minimum_decisive_pairs=1)["eligible_to_train"])
        with tempfile.TemporaryDirectory() as directory:
            split = create_session_held_out_split(rows, "2022-01-01", Path(directory) / "split.json")
        self.assertEqual(len(split["images"]["test"]), 2)
        self.assertTrue(all("2022-01-01" not in path for path in split["images"]["train"] + split["images"]["validation"]))

    def test_calibration_is_fitted_on_validation_records(self):
        calibration = fit_validation_calibration([
            {"winner": "1", "margin": 2., "reward_strength": 2.},
            {"winner": "2", "margin": -2., "reward_strength": 2.},
            {"winner": "tie", "margin": .01, "reward_strength": 1.},
            {"winner": "not_apply", "margin": 0., "reward_strength": -1.},
        ])
        self.assertEqual(calibration["fitted_on"], "validation_only")
        self.assertGreater(calibration["temperature"], 0.)

    def test_five_fold_test_pairs_have_a_held_out_endpoint(self):
        rows = [
            {"pair_id": str(index), "left": f"/images/{index}.bmp", "right": f"/images/{index + 1}.bmp", "winner": "1", "reconstruction_type": "HTR"}
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            protocol = create_fixed_five_fold_unseen_image_splits(rows, Path(directory) / "split.json")
        self.assertEqual(len(protocol["folds"]), 5)
        for fold in protocol["folds"]:
            test_images = set(fold["images"]["test"])
            self.assertTrue(all(any(image in test_images for image in (rows[int(pair_id)]["left"], rows[int(pair_id)]["right"])) for pair_id in fold["pair_ids_by_role"]["test"]))
            self.assertTrue(set(fold["images"]["test"]).isdisjoint(fold["images"]["train"]))

if __name__ == "__main__": unittest.main()

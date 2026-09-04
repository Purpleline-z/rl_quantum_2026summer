import unittest
import torch

from paper_replicate.peak_aware_static_rheed_reward_model import PeakAwareStaticRHEEDRewardModel, RHEEDPeakFeatureExtractor

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

if __name__ == "__main__": unittest.main()

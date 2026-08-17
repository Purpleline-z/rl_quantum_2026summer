import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from image_metadata_and_symmetry_features import fit_metadata_model, load_metadata, metadata_vector_for_path, mixture_metrics, symmetry_image
from image_mixture_features import fit_reference_nmf, score_nmf_features


class AdvancedFeatureTests(unittest.TestCase):
    def test_left_mirror_replaces_right_half(self):
        image = Image.fromarray(np.array([[1, 2, 8, 9]], dtype=np.uint8))
        self.assertEqual(np.asarray(symmetry_image(image, "left_half_mirror")).tolist(), [[1, 2, 2, 1]])

    def test_symmetric_average_is_symmetric(self):
        image = Image.fromarray(np.array([[0, 10, 30, 40]], dtype=np.uint8))
        result = np.asarray(symmetry_image(image, "symmetric_average"))
        self.assertTrue(np.array_equal(result, np.fliplr(result)))

    def test_mixture_entropy_orders_ambiguous_predictions_higher(self):
        metrics = mixture_metrics(np.array([[10., 0., 0.], [1., 1., 1.]]))
        self.assertGreater(metrics["mixture_entropy"][1], metrics["mixture_entropy"][0])
        self.assertTrue(np.allclose(metrics["probabilities"].sum(axis=1), 1))

    def test_metadata_join_and_train_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            pd.DataFrame({"path": ["a.bmp", "b.bmp", "extra.bmp"], "temperature": [10, 12, 100]}).to_csv(path, index=False)
            matched = load_metadata(path, ["a.bmp", "b.bmp"])
            model = fit_metadata_model(matched.iloc[:1], ["temperature"])
            self.assertEqual(len(matched), 2)
            self.assertGreater(model.score(np.array([[12.]]))[0], 0)
            vector = metadata_vector_for_path(matched, model, "unseen.bmp")
            self.assertEqual(vector.shape, (2,))  # value plus missingness flag

    def test_feature_nmf_refuses_signed_features_and_scores_nonnegative_features(self):
        labels = ["a", "a", "b", "b"]
        features = np.array([[1., 0., 1., 0.], [2., .1, .8, 0.], [0., 1., 0., 1.], [.1, 2., 0., .8]])
        diagnostic, _ = fit_reference_nmf(features, labels, n_components=2)
        scores = score_nmf_features(diagnostic, features[:2])
        self.assertEqual(scores["weights"].shape, (2, 2))
        self.assertTrue(np.allclose(scores["weights"].sum(axis=1), 1))
        with self.assertRaises(ValueError):
            fit_reference_nmf(np.array([[-1., 1.], [1., 1.]]), ["a", "b"], n_components=2)


if __name__ == "__main__":
    unittest.main()

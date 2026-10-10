"""Classification baselines must use the paper's split, fit on references only, and never touch test labels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY = Path(__file__).resolve().parents[2] / "active_learning_studies" / "classification_baselines"
sys.path.insert(0, str(STUDY)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_features as feats  # noqa: E402
import baseline_protocol as proto  # noqa: E402

PAPER_NN = Path(__file__).resolve().parents[2] / "active_learning_studies/pair_disjoint_not_image_disjoint/results/simclr_three_seed_identity_safe_task3/frozen_encoder_nearest_neighbour_baseline.csv"


@pytest.mark.parametrize("seed", proto.PAPER_SEEDS)
def test_split_is_the_papers_and_identity_disjoint(seed):
    split = proto.load_split(seed)
    proto.assert_no_overlap(split)  # raises on any shared SHA-256 content identity
    assert sum(len(v) for v in split["test"].values()) == 28
    assert set(split["references"]) == set(proto.CLASSES)   # Twinned excluded, as in the paper


def test_overlap_check_fires():
    split = {"identity": {"references": {"a": "h1"}, "test": {"b": "h1"}, "utility": {"c": "h3"}}}
    with pytest.raises(AssertionError):
        proto.assert_no_overlap(split)


def test_simclr_1nn_reproduces_the_committed_paper_baseline():
    """Same split + same encoder + same rule must give the numbers already in the repository, seed by seed."""
    expected = pd.read_csv(PAPER_NN).set_index("seed").outer_test_accuracy
    for seed in proto.PAPER_SEEDS:
        split = proto.load_split(seed)
        rp, ry = proto.flatten(split["references"]); tp, ty = proto.flatten(split["test"])
        rx, tx = proto.prepare("deep", feats.cached_features("simclr_resnet18", rp), feats.cached_features("simclr_resnet18", tp))
        assert proto.score(ty, proto.predict_knn(rx, ry, tx, 1))["accuracy"] == pytest.approx(expected[seed])


def test_knn_nearest_and_tie_break():
    ref = np.array([[1., 0], [0, 1.], [0.9, 0.1]]); ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    labels = np.array(["a", "b", "b"])
    x = np.array([[1., 0.05]]); x /= np.linalg.norm(x)
    assert proto.predict_knn(ref, labels, x, 1)[0] == "a"
    assert proto.predict_knn(ref, labels, x, 3)[0] == "b"          # majority of three
    # 2-NN tie (one vote each): the class with the larger summed similarity wins
    assert proto.predict_knn(ref, labels, x, 2)[0] == "a"


def test_tabular_standardisation_uses_reference_statistics_only():
    rng = np.random.default_rng(0); ref, test = rng.normal(size=(30, 6)), rng.normal(size=(8, 6))
    a = proto.prepare("tabular", ref, test)[0]; b = proto.prepare("tabular", ref, test * 100 + 7)[0]
    np.testing.assert_allclose(a, b)  # changing the test features cannot change how references are transformed


def test_logreg_depends_only_on_references_and_is_row_order_equivariant():
    rng = np.random.default_rng(1)
    centres = {c: rng.normal(size=12) * 3 for c in proto.CLASSES}
    ref_y = np.array([c for c in proto.CLASSES for _ in range(15)]); ref_x = np.stack([centres[c] + rng.normal(size=12) for c in ref_y])
    test_y = np.array(list(proto.CLASSES) * 3); test_x = np.stack([centres[c] + rng.normal(size=12) for c in test_y])
    pred = proto.predict_logreg(ref_x, ref_y, test_x, seed=0)
    assert (pred == test_y).mean() > .9
    perm = rng.permutation(len(test_x))
    assert (proto.predict_logreg(ref_x, ref_y, test_x[perm], seed=0) == pred[perm]).all()
    # the signature takes no test labels at all, and the hyper-parameter search sees only the references
    import inspect
    assert "test_y" not in inspect.signature(proto.predict_logreg).parameters
    assert "test_y" not in inspect.signature(proto.choose_c_by_reference_cv).parameters


def test_score_known_confusion():
    truth = np.array(["HTR", "HTR", "(1 x 1)", "(1 x 1)", "c(6 x 2)", "(√13 x √13)"])
    pred = np.array(["HTR", "(1 x 1)", "(1 x 1)", "(1 x 1)", "c(6 x 2)", "(√13 x √13)"])
    row = proto.score(truth, pred)
    assert row["n_test"] == 6 and row["n_correct"] == 5 and row["accuracy"] == pytest.approx(5 / 6)
    assert row["HTR_recall"] == pytest.approx(.5) and row["HTR_precision"] == pytest.approx(1.0)
    assert row["recall[(1 x 1)]"] == 1.0


def test_random_encoder_is_deterministic_per_seed_and_512_dimensional(tmp_path, monkeypatch):
    from PIL import Image
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"; Image.fromarray((np.random.default_rng(i).random((40, 50)) * 255).astype("uint8")).save(p); paths.append(p)
    a = feats.encode(feats.resnet18_encoder("random", 0), paths, feats.paper_transform())
    b = feats.encode(feats.resnet18_encoder("random", 0), paths, feats.paper_transform())
    c = feats.encode(feats.resnet18_encoder("random", 1), paths, feats.paper_transform())
    assert a.shape == (3, 512) and np.allclose(a, b) and not np.allclose(a, c)
    assert feats.raw_pixels(paths).shape == (3, 64 * 64) and feats.peak_profile(paths).shape == (3, 24)


def test_unavailable_weights_are_reported_not_substituted(monkeypatch):
    def refuse(*args, **kwargs): raise OSError("host blocked")
    monkeypatch.setattr(feats.models, "resnet50", refuse)
    with pytest.raises(feats.WeightsUnavailable):
        feats.torchvision_imagenet("resnet50")


def test_finetune_epoch_is_selected_on_utility_validation_with_earliest_tie():
    import finetune_baseline as ft
    assert ft.select_epoch([.5, .8, .8, .7]) == 2     # ties -> earliest epoch
    assert ft.select_epoch([.9]) == 1
    import inspect
    assert "test_y" not in inspect.signature(ft.select_epoch).parameters

"""Graph-aware strategies: valid picks, reproducibility, label-free graph built only from the images a selector can see, controls differ from the real rules."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

STUDY = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"
sys.path.insert(0, str(STUDY)); sys.path.insert(0, str(Path(__file__).parent))
import graph_strategies as gs  # noqa: E402
import judgment_unit_strategies as ju  # noqa: E402
from test_judgment_unit import _Model, _judgments  # noqa: E402


@pytest.mark.parametrize("name", gs.GRAPH_NAMES)
def test_selectors_return_budget_distinct_judgments_reproducibly(name):
    cands, labeled, cache = _judgments(); select = ju.make_selector(name, None)
    first = [x["pair_id"] for x in select(cands, labeled, _Model(), cache, 12, 3)]; second = [x["pair_id"] for x in select(cands, labeled, _Model(), cache, 12, 3)]
    assert first == second and len(first) == 12 and len(set(first)) == 12 and set(first) <= {x["pair_id"] for x in cands}


@pytest.mark.parametrize("name", sorted(gs.PAIR_RULES))
def test_pair_level_graph_rule_reveals_one_judgment_per_pair(name):
    cands, labeled, cache = _judgments(); chosen = ju.make_selector(name, None)(cands, labeled, _Model(), cache, 12, 3)
    assert len({x["base_pair"] for x in chosen}) == 12


def test_knn_graph_is_symmetric_without_self_loops_and_pagerank_is_a_distribution():
    x = np.random.default_rng(0).standard_normal((40, 8)); a = gs.knn_graph(x, 5)
    assert np.array_equal(a, a.T) and a.diagonal().sum() == 0 and a.sum(1).min() >= 5
    r = gs.pagerank(a); assert abs(r.sum() - 1) < 1e-9 and (r > 0).all()


def test_propagation_with_zero_steps_is_identity_and_smooths_otherwise():
    x = np.random.default_rng(1).standard_normal((30, 4)); a = gs.knn_graph(x, 4)
    assert np.allclose(gs.propagate(a, x, 0), x) and gs.propagate(a, x, 3).std() < x.std()


def test_percentile_ties_share_a_rank():
    p = gs._percentile(np.array([1., 1., 3., 2.])); assert p[0] == p[1] and p.max() == 1.0 and p[2] > p[3] > p[0]


def test_graph_nodes_are_only_images_of_candidates_and_labeled():
    cands, labeled, cache = _judgments(); cache = dict(cache); cache["held_out_image"] = torch.randn(512)
    scores = gs.node_scores("centrality", cands, labeled, cache, 0, False)
    assert "held_out_image" not in scores and set(scores) == {p for x in cands + labeled for p in (x["img1"], x["img2"])}


def test_graph_scores_do_not_depend_on_candidate_order():
    cands, labeled, cache = _judgments(); a = gs.node_scores("bridge", cands, labeled, cache, 5, False); b = gs.node_scores("bridge", cands[::-1], labeled[::-1], cache, 5, False)
    assert a == b


def test_shuffled_control_differs_but_keeps_the_score_distribution():
    cands, labeled, cache = _judgments(); real = gs.node_scores("centrality", cands, labeled, cache, 2, False); fake = gs.node_scores("centrality", cands, labeled, cache, 2, True)
    assert real != fake and sorted(real.values()) == sorted(fake.values())


DATA = STUDY.parents[2] / "data" / "original data"


@pytest.mark.skipif(not DATA.exists() or not (STUDY / "results" / "frozen_encoder_task3" / "simclr_feature_cache.pt").exists(), reason="needs the data and the feature cache")
@pytest.mark.parametrize("split", ["A", "B"])
def test_graph_strategies_obey_the_judgment_unit_invariants_on_real_data(split, tmp_path):
    import frozen_encoder_reward_head as frozen
    import judgment_unit_study as study
    import run_pair_endpoint_study as single
    torch.set_num_threads(2); cache = frozen.load_feature_cache(single.CACHE, single.DATA); ctx = study.Context(42, tmp_path, cache, split, "groups", .01, 20)
    model = ctx.fit(ctx.initial); state = {"cands": study.make_candidates(ctx, ctx.pool, ctx.initial, model, 42)}
    for name in gs.GRAPH_NAMES:
        picks = study.choose(ctx, name, ju.make_selector(name, ctx.exp), ctx.pool, ctx.initial, model, 10, 42, state)
        assert len(set(picks)) == 10 and set(picks) <= set(ctx.pool), name
        assert len(study.rows_of(ctx.exp, ctx.initial + picks)) == len(ctx.initial) + 10, name

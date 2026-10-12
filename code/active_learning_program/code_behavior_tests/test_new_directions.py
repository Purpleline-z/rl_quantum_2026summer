"""Integrity tests for the new-directions screens: seed disjointness, no test labels or hidden outcomes in fitting/selection, mirror head invariance, GP/Bayes sanity."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists(), reason="feature cache not built (build_feature_cache.py)")


@pytest.fixture(scope="module")
def ctx_b():
    import nd_select
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(2000, "B", scratch, core.feature_cache()); nd_select.CTX_REF["ctx"] = ctx; yield ctx


def test_seed_sets_are_disjoint_from_earlier_work():
    assert not (set(core.DEV_SEEDS) | set(core.CONFIRM_SEEDS)) & core.USED_SEEDS_BY_EARLIER_WORK
    assert not set(core.DEV_SEEDS) & set(core.CONFIRM_SEEDS)


def test_random_label_sets_are_shared_and_exclude_heldout(ctx_b):
    a, b = core.random_labelled(ctx_b, 20, 1), core.random_labelled(ctx_b, 20, 1)
    assert a == b and len(a) == len(ctx_b.initial) + 20 and not set(a[len(ctx_b.initial):]) & set(ctx_b.initial)
    held = {x for g in ctx_b.test for r in ctx_b.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    assert not held & {ctx_b.info[j][k] for j in a for k in ("img1", "img2")}


def test_predictions_do_not_depend_on_test_labels(ctx_b):
    import nd_learners as L
    test = core.TestSet(ctx_b); lab = core.random_labelled(ctx_b, 10, 0); predictor = L.baseline(ctx_b, lab, 0)
    d1 = predictor(test); test.sign = -test.sign; test.weight = test.weight * 0 + 1
    assert np.allclose(d1, predictor(test))


def test_mirror_head_is_exactly_invariant_to_the_mirror(ctx_b):
    import nd_learners as L
    head = L.SymmetricHead(core.new_head(512, 0)).eval(); x, m = torch.randn(5, 512), torch.randn(5, 512)
    assert torch.allclose(head(torch.cat([x, m], 1)), head(torch.cat([m, x], 1)), atol=1e-6)


def test_selectors_ignore_hidden_outcomes_of_candidates(ctx_b):
    import judgment_unit_study as ju_study
    import nd_select
    import new_methods_strategies as nm
    nm.CTX["ctx"] = ctx_b; model = ctx_b.fit(ctx_b.initial); cands, labeled, cache = ju_study.make_candidates(ctx_b, ctx_b.pool, ctx_b.initial, model, 0)
    stripped = [{k: v for k, v in c.items() if k != "outcomes"} for c in cands]
    for name in ("vopt_hidden", "vopt_raw", "consistency", "vopt_amb", "gp_var"):
        a = [x["pair_id"] for x in nd_select.SELECTORS[name](cands, labeled, model, cache, 10, 0)]
        b = [x["pair_id"] for x in nd_select.SELECTORS[name](stripped, labeled, model, cache, 10, 0)]
        assert a == b, name


def test_selection_returns_distinct_pool_members(ctx_b):
    import judgment_unit_study as ju_study
    import nd_select
    model = ctx_b.fit(ctx_b.initial); cands, labeled, cache = ju_study.make_candidates(ctx_b, ctx_b.pool, ctx_b.initial, model, 0)
    ids = {c["pair_id"] for c in cands}
    for name in ("vopt_hidden", "vopt_raw", "vopt_stoch3", "consistency", "vopt_cons", "gp_var"):
        picks = [x["pair_id"] for x in nd_select.SELECTORS[name](cands, labeled, model, cache, 20, 0)]
        assert len(picks) == 20 == len(set(picks)) and set(picks) <= ids, name


def test_stochastic_batch_is_reproducible_per_seed(ctx_b):
    import judgment_unit_study as ju_study
    import nd_select
    model = ctx_b.fit(ctx_b.initial); cands, labeled, cache = ju_study.make_candidates(ctx_b, ctx_b.pool, ctx_b.initial, model, 0)
    a = [x["pair_id"] for x in nd_select.SELECTORS["vopt_stoch3"](cands, labeled, model, cache, 10, 3)]; b = [x["pair_id"] for x in nd_select.SELECTORS["vopt_stoch3"](cands, labeled, model, cache, 10, 3)]
    c = [x["pair_id"] for x in nd_select.SELECTORS["vopt_stoch3"](cands, labeled, model, cache, 10, 4)]
    assert a == b and a != c


def test_gp_and_bayes_predictors_run_and_are_antisymmetric(ctx_b):
    import nd_bayes
    import nd_gp
    test = core.TestSet(ctx_b); lab = core.random_labelled(ctx_b, 10, 0); swapped = core.TestSet(ctx_b); swapped.img1, swapped.img2 = test.img2, test.img1
    for predictor in (nd_gp.fit_gp(ctx_b, lab, 0, tune_length=False)[0], nd_bayes.fit_bayes(ctx_b, lab, 0, False)[0]):
        d = np.asarray(predictor(test)); assert np.isfinite(d).all() and np.allclose(d, -np.asarray(predictor(swapped)), atol=1e-5)

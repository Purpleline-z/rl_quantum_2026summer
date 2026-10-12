"""Graph-selection screen (Prompt 2): seed registry, valid and reproducible picks on a real context, graph built only from visible images, controls differ from the rules, analysis helpers."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402
import gsel_analyze as ga  # noqa: E402

needs_cache = pytest.mark.skipif(not core.single_study.CACHE.exists(), reason="feature cache not built (build_feature_cache.py)")


def test_new_seed_ranges_are_registered_as_used_and_disjoint_from_dev_and_confirm_sets():
    used = core.USED_SEEDS_BY_EARLIER_WORK
    assert set(range(8000, 8400)) <= used and set(range(1400, 1740)) <= used
    assert not (set(core.DEV_SEEDS) | set(core.CONFIRM_SEEDS)) & used


def test_holm_is_monotone_and_bounded():
    p = {"a": .001, "b": .01, "c": .04, "d": .5}; h = ga.holm(p)
    assert h["a"] == pytest.approx(.004) and h["b"] == pytest.approx(.03) and h["c"] == pytest.approx(.08) and h["d"] == pytest.approx(.5)
    assert all(0 <= v <= 1 for v in h.values())


def _frame():
    rows = []
    for seed in range(8):
        for budget in (10, 20):
            for sel, off in (("random", 0.0), ("good", .05), ("bad", -.05)):
                for draw in ((0, 1) if sel == "random" else (0,)):
                    base = .6 + .01 * seed
                    rows.append({"seed": seed, "split": "A", "budget": budget, "selector": sel, "draw": draw, "learner": "aw8_all", "acc": base + off, "auc": base + off, "ll": .5 - off, "htr_acc": base + off, "htr_auc": base + off,
                                 "htr_ll": .5 - off, "t1x1_acc": base, "tc6x2_acc": base, "t13_acc": base, "htr_acc_": 0, "pick_t1x1": 2, "pick_tc6x2": 2, "pick_t13": 3, "pick_htr": 3, "pick_images": 12, "pick_decisive": 5})
    f = pd.DataFrame(rows); f["other_acc"] = f[list(ga.OTHER)].mean(axis=1); return f


def test_gain_signs_log_loss_is_flipped_and_other_acc_is_type_mean():
    f = _frame(); good = ga.gains(f, "good"); bad = ga.gains(f, "bad")
    assert (good.acc > 0).all() and (good.ll > 0).all() and (bad.acc < 0).all() and (bad.ll < 0).all()
    assert np.allclose(good.acc, .05) and np.allclose(good.other_acc, 0)


def test_table_and_control_difference_and_composition():
    f = _frame(); t = ga.table(f, ["good", "bad"], metrics=("acc", "auc", "ll")); assert set(t.selector) == {"good", "bad"} and (t.n_seeds == 8).all()
    d = ga.vs_control(f, "good", "bad"); assert np.allclose(d[d.metric == "acc"]["diff"], .10)
    c = ga.composition(f, ["good"]); assert np.allclose(c.share_htr, .3) and np.allclose(c.decisive_share, .5) and "|" in ga.md(t)


@pytest.fixture(scope="module")
def real():
    import nd_select, gsel_select
    import judgment_unit_study as ju_study, new_methods_strategies as nm
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        cache = core.feature_cache(); ctx = core.make_ctx(2000, "B", scratch, cache); nd_select.CTX_REF["ctx"] = ctx; nm.CTX["ctx"] = ctx; model = ctx.fit(ctx.initial)
        cands, labeled, ccache = ju_study.make_candidates(ctx, ctx.pool, ctx.initial, model, 2000); yield gsel_select, ctx, model, cands, labeled, ccache


MODEL_FREE = ["tdc", "tdc_typeonly", "tdc_shuffled", "tcu", "htr_amb", "htr_amb_shuffled", "htr_bridge", "htr_bridge_shuffled", "htr_cov_amb", "htr_cov_amb_shuffled", "htr_cov_bridge", "htr_cov_bridge_shuffled"]


@needs_cache
@pytest.mark.parametrize("name", MODEL_FREE)
def test_graph_rules_return_budget_distinct_candidates_reproducibly(real, name):
    gsel, ctx, model, cands, labeled, cache = real; sel = gsel.REGISTRY[name]
    a = [x["pair_id"] for x in sel(cands, labeled, model, cache, 15, 3)]; b = [x["pair_id"] for x in sel(cands, labeled, model, cache, 15, 3)]
    assert a == b and len(a) == 15 and len(set(a)) == 15 and set(a) <= {x["pair_id"] for x in cands}


@needs_cache
def test_graph_nodes_are_only_visible_images_and_references_no_test_or_validation_images(real):
    gsel, ctx, model, cands, labeled, cache = real; import graph_typed as gt
    g = gt.ImageGraph(cands, labeled, cache, gt.reference_types(ctx.exp), 3)
    visible = {p for x in cands + labeled for p in (x["img1"], x["img2"])} | set(gt.reference_types(ctx.exp)); test = core.TestSet(ctx)
    assert set(g.paths) <= visible and not (set(test.img1) | set(test.img2)) & set(g.paths)


@needs_cache
def test_ambiguity_and_bridge_scores_are_in_range_and_shuffled_graph_changes_the_picks(real):
    gsel, ctx, model, cands, labeled, cache = real; import graph_typed as gt
    g = gt.ImageGraph(cands, labeled, cache, gt.reference_types(ctx.exp), 3); amb = gsel.htr_rt13_ambiguity(g); br = gsel.bridge_mass(g)
    assert (amb >= 0).all() and (amb <= 1 + 1e-9).all() and (br >= 0).all() and (br <= .5 + 1e-9).all()
    real_pick = {x["pair_id"] for x in gsel.REGISTRY["htr_cov_amb"](cands, labeled, model, cache, 20, 3)}; shuf = {x["pair_id"] for x in gsel.REGISTRY["htr_cov_amb_shuffled"](cands, labeled, model, cache, 20, 3)}
    assert real_pick != shuf


@needs_cache
def test_htr_priority_rules_only_pick_htr_or_rt13_for_ranking_rules(real):
    gsel, ctx, model, cands, labeled, cache = real
    for name in ("htr_amb", "htr_bridge"):
        picks = gsel.REGISTRY[name](cands, labeled, model, cache, 10, 3); assert all(int(x["type_idx"]) in (3, 4) for x in picks)

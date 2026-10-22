"""Tests of the V1a code (run: python3 -m pytest v1a/test_v1a.py -q).  Needs the feature caches (build_feature_cache.py, nd_temporal.py build)."""
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1a_core as V  # noqa: E402

pytestmark = pytest.mark.skipif(not all(p.exists() for p in V.CACHE_FILES), reason="feature caches not built")


@pytest.fixture(scope="module")
def ctx():
    torch.set_num_threads(1); cache = V.load_cache()
    with tempfile.TemporaryDirectory() as scratch: yield V.V1aContext(42, scratch, cache, 3e-3, 100)


def test_head_has_132613_parameters_and_four_active_types(ctx):
    assert sum(p.numel() for p in ctx.features.base_model.reward_head.parameters()) == 132613 and set(ctx.exp.references) == set(V.ACTIVE)


def test_ideal_split_is_88_28_28_and_bad_images_18(ctx):
    assert sum(map(len, ctx.exp.references.values())) == 88 and sum(map(len, ctx.exp.utility_images.values())) == 28 and sum(map(len, ctx.exp.test_images.values())) == 28 and len(ctx.exp.bad_paths) == 18
    assert not {str(p) for ps in ctx.exp.references.values() for p in ps} & {str(p) for ps in ctx.exp.test_images.values() for p in ps}


def test_trajectory_set_has_no_byte_identical_ideal_image(ctx):
    ideal = {V.sha256_of(p) for p in V.all_ideal_files()}; keep, info = V.trajectory_set(ctx.features.cache)
    assert not ideal & {V.sha256_of(p) for p in keep} and info["T"] == len(keep) == info["n_trajectory_images"] - info["n_removed_identical_to_ideal"] and all("Trajectories" in p for p in keep)


def test_pair_term_equals_the_repository_fit_head_term(ctx):
    rows = V.ju_study.rows_of(ctx.exp, ctx.initial); head = ctx.features.base_model.reward_head.eval(); typ = torch.as_tensor(rows.type_idx.to_numpy()); idx = torch.arange(len(rows))
    xa, xb = ctx.features.get(rows.resolved_img1), ctx.features.get(rows.resolved_img2); w = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32); winner = rows.Winner.to_numpy()
    a, b = head(xa)[idx, typ], head(xb)[idx, typ]; masks = {l: torch.as_tensor(winner == l) for l in V.frozen.LABELS}
    ref = ((-F.logsigmoid(a - b) * masks["1"] - F.logsigmoid(b - a) * masks["2"] + (a - b).abs() * masks["tie"] + (F.relu(a) + F.relu(b)) * masks["not_apply"]) * w).sum() / len(rows)
    assert torch.allclose(V.pair_loss(a, b, winner, w), ref)


def test_ideal_and_bad_expectation_equal_classifier2_sampling_in_the_limit(ctx):
    """Monte Carlo of train_unified._get_ideal_sample / _get_bad_sample (type uniform, image uniform, 70 % trajectory / 30 % uniform other type + uniform image) vs the closed-form expectation."""
    torch.manual_seed(0); rng = random.Random(0); head = ctx.features.base_model.reward_head.eval(); refs = {t: head(ctx.features.get(ctx.exp.references[t])).detach() for t in V.ACTIVE}
    sT = head(ctx.traj).detach(); sB = head(ctx.features.get(ctx.exp.bad_paths)).detach(); exact = float(V.ideal_loss(refs, sT)); total = 0.0; n = 60000
    for _ in range(n):
        t = rng.choice(V.ACTIVE); c = V.COL[t]; x = refs[t][rng.randrange(len(refs[t])), c]
        if rng.random() < V.IDEAL_VS_TRAJECTORY: other = sT[rng.randrange(len(sT)), c]
        else: o = rng.choice([u for u in V.ACTIVE if u != t]); other = refs[o][rng.randrange(len(refs[o])), c]
        total += float(F.softplus(other - x))
    assert abs(total / n - exact) < 0.02 * max(exact, 1e-6) + 0.01
    total = 0.0
    for _ in range(n):
        tau = rng.randrange(5); v = sB[rng.randrange(len(sB)), tau]; u = sT[rng.randrange(len(sT)), tau]; total += float(F.softplus(v - u))
    assert abs(total / n - float(V.bad_loss(sB, sT))) < 0.02 * float(V.bad_loss(sB, sT)) + 0.01


def test_win_rate_accuracy_equals_the_repository_evaluate_model(ctx):
    m = ctx.fit(ctx.initial, lr=3e-3, steps=20); truth, pred = V.win_rate_predict(ctx.exp, ctx.features, m); acc, _ = V.accuracy_and_macro_f1(truth, pred)
    assert abs(acc - V.frozen.evaluate_model(ctx.exp, ctx.features, m)["test_accuracy"]) < 1e-12 and len(truth) == 28
    vt, vp = V.win_rate_predict(ctx.exp, ctx.features, m, "utility"); assert abs(V.accuracy_and_macro_f1(vt, vp)[0] - V.frozen.evaluate_model(ctx.exp, ctx.features, m, "utility")["test_accuracy"]) < 1e-12


def test_same_initial_head_at_every_retrain_and_test_images_never_trained_on(ctx):
    m0, m1 = ctx.fit(ctx.initial, lr=1e-3, steps=0), ctx.fit(ctx.initial + ctx.pool[:10], lr=1e-3, steps=0)
    assert all(torch.equal(a, b) for a, b in zip(m0.reward_head.parameters(), m1.reward_head.parameters()))
    test_ideal = {str(p) for ps in ctx.exp.test_images.values() for p in ps}; used = {str(p) for ps in ctx.exp.references.values() for p in ps} | {str(p) for ps in ctx.exp.utility_images.values() for p in ps}
    assert not test_ideal & used and not test_ideal & {str(x) for x in ctx.exp.bad_paths}
    keep, _ = V.trajectory_set(ctx.features.cache); assert not test_ideal & set(keep)

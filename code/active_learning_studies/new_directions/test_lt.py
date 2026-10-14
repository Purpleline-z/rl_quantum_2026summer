"""Unit tests of the latent-type study.  python3 test_lt.py  (needs PYTHONPATH=.:../active_learning_program)"""
import itertools, tempfile
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, lt_model as M, lt_run as R


def test_seed_blocks():
    assert not core.LT_SEEDS_ALL & (core.USED_SEEDS_BY_EARLIER_WORK - core.LT_SEEDS_ALL) and len(core.LT_SEEDS_ALL) == 90
    assert core.LT_DEV_SEEDS[0] == 10700 and core.LT_CONFIRM1[0] == 10730 and core.LT_CONFIRM2[-1] == 10799


def test_table():
    t = M.init_table(); assert np.allclose(t.sum(2), 1) and np.allclose(t[0, 1, [0, 1, 2, 3]], t[1, 0, [1, 0, 2, 3]]) and np.allclose(t[1, 1, 0], t[1, 1, 1])


def test_bp_matches_enumeration():
    rng = np.random.default_rng(0); prior = rng.dirichlet(np.ones(5), 3); fac = {(0, 1): [(0, 0), (2, 3)], (1, 2): [(1, 2)]}; table = M.init_table()
    belief, pair = M.belief_propagation(prior, fac, table, iters=50)
    joint = np.zeros((5, 5, 5))
    for z in itertools.product(range(5), repeat=3):
        p = prior[0, z[0]] * prior[1, z[1]] * prior[2, z[2]]
        for (a, b), items in fac.items(): p *= M.factor_potential(items, table)[z[a], z[b]]
        joint[z] = p
    joint /= joint.sum()
    for i in range(3): assert np.allclose(belief[i], joint.sum(tuple(j for j in range(3) if j != i)), atol=1e-6), i   # tree: BP is exact
    assert np.allclose(pair[(0, 1)], joint.sum(2), atol=1e-6)


def test_mstep_normalised_and_modes():
    rng = np.random.default_rng(1); prior = rng.dirichlet(np.ones(5), 4); fac = {(0, 1): [(0, 0)], (2, 3): [(3, 1), (1, 2)]}
    class J: factors = fac; rows = []
    belief, table, _ = M.posterior_of_images(J, prior, "em"); assert np.allclose(table.sum(2), 1)
    b2, _, _ = M.posterior_of_images(J, prior, "prior"); assert np.allclose(b2, prior)
    b3, _, _ = M.posterior_of_images(J, prior, "random", rng=np.random.default_rng(3)); assert np.allclose(np.sort(b3, 0), np.sort(belief, 0), atol=1e-9)


def test_off_equals_baseline():
    cache = core.feature_cache(); torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as s:
        ctx = core.make_ctx(10701, "A", s, cache); test = core.TestSet(ctx); lab = core.random_labelled(ctx, 20, 0)
        d0 = R.REGISTRY["lap3_k10_st4"](ctx, lab, 3)(test); d1 = R.REGISTRY["lt_off"](ctx, lab, 3)(test); assert np.abs(d0 - d1).max() == 0.0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)

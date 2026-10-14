"""Physics-features study: seed block, 16-bit decoding, extractor on synthetic RHEED-like images, label-free standardisation (held-out images excluded), controls."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

READY = core.single_study.CACHE.exists() and (NEW / "results" / "pf" / "blocks_v1.pt").exists() and (NEW / "results" / "pf" / "f16.pt").exists()


def synthetic(a=62, fractions=(), seed=0, h=492, w=656, x0=295, row=225, edge=140):
    """Dark above the horizon, diffuse background below, streaks at +-a (integer order) and at the given fractions of a, a bright specular spot."""
    rng = np.random.default_rng(seed); g = np.full((h, w), 5.0, dtype=np.float32); g[edge:] = 40.0; xs = np.arange(w)
    for u, amp in [(0, 150.0), (1, 60.0)] + [(f, 25.0) for f in fractions]:
        for s in (-1, 1) if u else (1,): g[edge + 5:row + 150] += amp * np.exp(-0.5 * ((xs - (x0 + s * u * a)) / 3.0) ** 2)[None, :]
    yy, xx = np.mgrid[:h, :w]; g += 250 * np.exp(-0.5 * (((yy - row) / 6) ** 2 + ((xx - x0) / 6) ** 2)); return np.clip(g + rng.normal(0, 1.0, g.shape), 0, 255)


def test_seed_block_new_and_disjoint():
    assert core.PF_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and min(core.PF_DEV_SEEDS) == 10600 and max(core.PF_CONFIRM2) == 10699 and len(core.PF_SEEDS_ALL) == 90
    assert not core.PF_SEEDS_ALL & (core.USED_SEEDS_BY_EARLIER_WORK - core.PF_SEEDS_ALL)


def test_sixteen_bit_images_are_decoded_not_clipped(tmp_path):
    from PIL import Image
    import pf_extract as P
    a16 = (np.random.default_rng(0).uniform(0, 12000, (580, 780))).astype(np.uint16); Image.fromarray(a16).save(tmp_path / "x.png")
    assert Image.open(tmp_path / "x.png").mode == "I;16"; g, mode = P.load_gray(tmp_path / "x.png")
    assert mode == "I;16" and g.shape == (492, 656) and 10 < g.mean() < 30 and g.max() < 60       # value / 257, not clipped at 255 (clipping gives a saturated disc)


def test_extractor_finds_geometry_and_fractional_streaks():
    import pf_extract as P
    g = synthetic(a=62, fractions=(0.5,)); edge = P.shadow_edge(g); r, x0 = P.specular(g, edge)
    assert abs(edge - 140) <= 6 and abs(r - 225) <= 4 and abs(x0 - 295) <= 3
    ex = {"profiles": {b: P.profile(g, max(r + a0, edge + 3), r + a1, x0, P.HALF) for b, (a0, a1) in P.BANDS.items()}, "lvl_above": 5.0, "lvl_below": 40.0, "edge": edge, "row": r, "col": x0}
    assert abs(P.spacing(ex) - 62) <= 3
    d = P.descriptors({**ex, "mode": "P"}); i_half = list(P.FRACS).index(0.5); pk = d["peaks"][:len(P.FRACS)]       # band a block of the peak table
    assert pk[i_half] > 0.15 and pk[list(P.FRACS).index(1 / 3)] < 0.1                                               # the 1/2 streak is seen, 1/3 is not
    d0 = P.descriptors({**ex, "profiles": {b: P.profile(synthetic(a=62, fractions=()), max(r + a0, edge + 3), r + a1, x0, P.HALF) for b, (a0, a1) in P.BANDS.items()}, "mode": "P"})
    assert d0["peaks"][i_half] < 0.5 * pk[i_half]                                                                    # no fractional streak -> low value


def test_shift_robustness_of_descriptors():
    import pf_extract as P
    out = []
    for dx in (0, 7):
        g = synthetic(a=62, fractions=(0.5,), x0=295 + dx); e = P.shadow_edge(g); r, x0 = P.specular(g, e)
        ex = {"profiles": {b: P.profile(g, max(r + a0, e + 3), r + a1, x0, P.HALF) for b, (a0, a1) in P.BANDS.items()}, "lvl_above": 5.0, "lvl_below": 40.0, "edge": e, "row": r, "col": x0, "mode": "P"}; out.append(P.descriptors(ex)["uprof"])
    assert np.abs(out[0] - out[1]).max() < 0.2


@pytest.mark.skipif(not READY, reason="physics cache not built")
def test_physics_tables_use_only_allowed_nodes_and_are_scale_matched():
    import nd_graph_learner as G
    import nd_physics as NP
    import nd_representation as R
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(10600, "A", scratch, core.feature_cache())
    nodes = G.graph_for(ctx, 10).paths; held = {str(x) for g in list(ctx.validation) + list(ctx.test) for r in ctx.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}; assert not held & set(nodes)
    t = NP.ctx_rep_pf(ctx, "pfa"); x = torch.stack([t[p] for p in nodes]); ref = float(torch.stack([R._blocks()["s4"][p] for p in nodes]).norm(dim=1).mean())
    assert abs(float(x.norm(dim=1).mean()) / ref - 1) < 1e-3 and float(x.mean(0).abs().max()) < 0.2 * float(x.std(0).max())          # centred / scaled with node statistics
    s = NP.ctx_rep_pf(ctx, "pfas"); assert next(iter(s.values())).shape[0] == 136 + 512 and NP.ctx_rep_pf(ctx, "prd")[nodes[0]].shape[0] == 136 + 512
    f = NP.ctx_rep_pf(ctx, "f16"); white = [p for p in nodes if "RT13_1.png" in p or "HTR_1.png" in p]
    for p in white: assert float((f[p] - R._blocks()["s4"][p]).abs().max()) > 1.0                                                 # the corrected feature differs from the saturated-disc feature


@pytest.mark.skipif(not READY, reason="physics cache not built")
def test_registered_learners_train_and_shuffle_control_differs_and_base_unchanged():
    import nd_graph_learner as G
    import nd_physics as NP
    import nd_representation as R
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(10600, "A", scratch, core.feature_cache())
    lab = core.random_labelled(ctx, 10, 0); test = core.TestSet(ctx); d = {n: np.asarray(NP.LEARNERS[n](ctx, lab, 3)(test)) for n in ("pf_pfa_hs", "pf_pfa_gs", "pf_pfa_gs_shuf", "pf_q_pfa")}
    assert all(np.isfinite(v).all() and len(v) == len(test.k) for v in d.values()) and not np.allclose(d["pf_pfa_gs"], d["pf_pfa_gs_shuf"])
    assert np.allclose(R.LEARNERS["rp_base"](ctx, lab, 3)(test), G.LEARNERS["lap3_k10_st4"](ctx, lab, 3)(test), atol=1e-6)         # patched ctx_rep leaves the base representation untouched

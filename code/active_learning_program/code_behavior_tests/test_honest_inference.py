"""Tests of the honest-inference estimator (new_directions/hi_infer.py) on synthetic per-judgment predictions: estimator = nd_core's standard endpoint, exact additive decomposition, resampling with unit
multiplicity 1 reproduces the estimate, coverage of the group bootstrap under group heterogeneity (and the under-coverage of a seed-only bootstrap), sign-flip p-values."""
import sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(ROOT))
import hi_infer as H  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def make_world(root: Path, rng, n_groups=40, n_seeds=12, mu=0.0, tau=0.6, per_cell=12, draws=2, splits=("A", "B"), budgets=(10, 20)):
    """Synthetic world.  Group g has a fixed role s_g = +-1: the better version of the logit (shifted by tau towards the truth) goes to the candidate if s_g = +1 and to the reference if s_g = -1, so the
    gain is heterogeneous across groups and exactly mean-zero in expectation at mu = 0; mu shifts the candidate towards the truth for every group.  Every cell holds `per_cell` random groups x 2 judgments."""
    groups = [f"g{i}" for i in range(n_groups)]; s_g = rng.choice([-1.0, 1.0], n_groups); (root / "meta").mkdir(parents=True); (root / "pred").mkdir(); pd.DataFrame({"group": groups}).to_csv(root / "clusters.csv", index=False)
    for seed in range(n_seeds):
        for split in splits:
            gs = rng.choice(n_groups, per_cell, replace=False); rows = [(g, j) for g in gs for j in range(2)]; sign = rng.choice([-1.0, 1.0], len(rows)); role = np.array([s_g[g] for g, _ in rows])
            pd.DataFrame({"j": range(len(rows)), "group": [groups[g] for g, _ in rows], "k": 0, "sign": sign, "weight": 1.0, "anc_all_min": rng.uniform(0, 0.1, len(rows)), "anc_ref_min": 0.0}).to_csv(root / "meta" / f"seed{seed}_{split}.csv", index=False)
            out = []
            for budget in budgets:
                for draw in range(draws):
                    base = sign * 0.6 + rng.normal(size=len(rows)); cand = base + sign * (mu + tau * (role > 0)) + 0.3 * rng.normal(size=len(rows)); ref = base + sign * tau * (role < 0) + 0.3 * rng.normal(size=len(rows))
                    for name, d in (("ref", ref), ("cand", cand)): out.append(pd.DataFrame({"budget": budget, "draw": draw, "learner": name, "j": range(len(rows)), "d": d}))
            pd.concat(out).to_csv(root / "pred" / f"seed{seed}_{split}.csv.gz", index=False)
    return s_g


def test_estimator_equals_standard_endpoint():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); make_world(t, rng); blk = H.Block(t, clusters=t / "clusters.csv")
        for metric in ("auc", "acc", "ll"):
            des = H.Design(blk, "cand", "ref", metric); vals = []
            for c, (st, sp) in enumerate(zip(blk.cells.start, blk.cells.stop)):
                s, dc, dr = blk.sign[st:sp], blk.D["cand"][st:sp], blk.D["ref"][st:sp]
                f = {"auc": lambda d: roc_auc_score(s > 0, d), "acc": lambda d: float(((d * s) > 0).mean()), "ll": lambda d: -float(np.logaddexp(0, -s * d).mean())}[metric]
                vals.append(f(dc) - f(dr))
            assert abs(des.estimate() - np.mean(vals)) < 1e-12, metric
            assert np.allclose(des.point(), vals)
            assert abs(des.contributions().sum() - des.estimate()) < 1e-12, metric                        # exact additive decomposition
            assert abs(des.perm_contributions().sum() - des.estimate()) < 1e-12, metric
            X = np.ones((3, blk.n_groups)); assert np.allclose(np.nanmean(des.cell_values(X), axis=1), des.estimate())
            Xd = 2 * X; assert np.allclose(np.nanmean(des.cell_values(Xd), axis=1), des.estimate())         # multiplicity scales numerator and denominator alike


def test_group_bootstrap_covers_and_seed_bootstrap_does_not():
    rng = np.random.default_rng(7); cover_g = cover_s = 0; R = 60
    for r in range(R):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); make_world(t, rng, n_groups=30, n_seeds=30, mu=0.0, tau=1.0, per_cell=10, draws=1, budgets=(10,)); blk = H.Block(t, clusters=t / "clusters.csv"); des = H.Design(blk, "cand", "ref", "acc")
            lo, hi = H.ci(H.group_bootstrap(des, np.arange(blk.n_groups), B=300, seed=r)); s = H.seed_level(des, B=300, seed=r)
            cover_g += lo <= 0 <= hi; cover_s += s["seed_lo"] <= 0 <= s["seed_hi"]
    assert cover_g / R >= 0.88, cover_g / R                    # group bootstrap: about nominal (small-sample percentile intervals are slightly liberal)
    assert cover_s / R < cover_g / R                           # seed-level bootstrap ignores the fixed group set


def test_signflip_p_uniform_under_null_and_small_under_effect():
    rng = np.random.default_rng(3); ps = []
    for r in range(80):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); make_world(t, rng, n_groups=30, n_seeds=10, mu=0.0, tau=1.0, per_cell=10, draws=1, budgets=(10,)); blk = H.Block(t, clusters=t / "clusters.csv"); des = H.Design(blk, "cand", "ref", "auc")
            ps.append(H.signflip_p(des.perm_contributions(), np.arange(blk.n_groups), n=2000, seed=r)[0])
    ps = np.array(ps); assert 0.02 <= (ps < 0.05).mean() <= 0.12, (ps < 0.05).mean()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); make_world(t, rng, n_groups=60, n_seeds=20, mu=0.8, tau=0.3, per_cell=15, draws=1, budgets=(10,)); blk = H.Block(t, clusters=t / "clusters.csv"); des = H.Design(blk, "cand", "ref", "auc")
        assert H.signflip_p(des.perm_contributions(), np.arange(blk.n_groups), n=4000)[0] < 0.01

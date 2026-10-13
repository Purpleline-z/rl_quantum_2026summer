"""Statistics helpers of the recipe study on synthetic result files (no feature cache needed)."""
import json
import sys
from pathlib import Path

import numpy as np

GE = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint" / "graph_exploration"
sys.path.insert(0, str(GE))
import rc_confirm  # noqa: E402
import rc_tables as T  # noqa: E402


def make_dir(tmp_path, gain_auc=0.02, n_seeds=12):
    rng = np.random.default_rng(1); d = tmp_path / "cells"; d.mkdir()
    for seed in range(n_seeds):
        for split in "AB":
            recs = []; anc = [0.0005, 0.0005, 0.05, 0.05, 0.2, 0.0005, 0.05, 0.2]; sign = [1, -1, 1, -1, 1, 1, -1, -1]
            for budget in (10, 20):
                for draw in range(2):
                    base = dict(seed=seed, split=split, budget=budget, draw=draw, auc=0.7 + 0.01 * rng.normal(), acc=0.65 + 0.01 * rng.normal(), ll=0.6, htr_acc=0.5, htr_ll=0.7, far_auc=0.6, far_acc=0.6)
                    recs.append({**base, "learner": "ref", "d": [0.0] * 8})
                    recs.append({**base, "learner": "cand", "auc": base["auc"] + gain_auc, "acc": base["acc"] + gain_auc, "ll": 0.5, "d": [0.1] * 8})
            (d / f"seed{seed}_{split}.json").write_text(json.dumps({"meta": {"anc_min": anc, "sign": sign, "weight": [1] * 8, "k": [0] * 8}, "records": recs}))
    return d


def test_holm_matches_hand_computation():
    assert np.allclose(T.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_gain_signs_and_pairing(tmp_path):
    df = T.load([str(make_dir(tmp_path))]); s = T.per_seed_gain(df, "cand", "ref", "A", "auc"); assert np.allclose(s, 0.02) and len(s) == 12
    assert np.allclose(T.per_seed_gain(df, "cand", "ref", "A", "ll"), 0.1)       # log-loss: reference minus candidate


def test_confirmation_verdict_flags_a_clear_gain_and_no_gain(tmp_path):
    d = make_dir(tmp_path); tab = rc_confirm.family_table(T.load([str(d)]), "ref", ["cand"], ("auc", "acc")); v = rc_confirm.verdicts(tab)
    assert bool(v.SUCCESS.iloc[0]) and (tab.gain > 0).all()


def test_strata_use_the_stored_anchor_distance(tmp_path):
    dfd = T.load([str(make_dir(tmp_path))], keep_d=True); tab = T.strata_table(dfd, "ref", ["cand"]); assert set(tab.stratum) >= {"near (<=0.001)", "far (>0.03)"}

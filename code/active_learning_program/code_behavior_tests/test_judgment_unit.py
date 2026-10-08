"""Judgment-unit study: one (pair, type) judgment is the query unit.

* training on judgment rows uses exactly those rows (an unselected judgment of a selected pair is NOT used);
* every selector returns ``budget`` distinct judgments drawn from the candidates, reproducibly;
* pair-level rules reveal one judgment per chosen pair;
* (integration, real data) every strategy of the study obeys the same invariants on a real split, and the validation/test images never reach the labelled set or the pool.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

STUDY = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"
sys.path.insert(0, str(STUDY))
import frozen_encoder_reward_head as frozen  # noqa: E402
import judgment_unit_strategies as ju  # noqa: E402


class _Model:
    def __init__(self):
        torch.manual_seed(0); self.reward_head = nn.Sequential(nn.Linear(512, 16), nn.ReLU(), nn.Dropout(.2), nn.Linear(16, 5))


def _judgments(n_pairs=30, labeled_pairs=6):
    """Synthetic judgment candidates: several types per pair, shared image pool."""
    g = torch.Generator().manual_seed(1); cache = {f"i{k}": torch.randn(512, generator=g) for k in range(80)}
    types = (0, 2, 3, 4)
    def make(pair, t, first):
        return {"pair_id": f"p{pair}#{t}", "base_pair": f"p{pair}", "img1": f"i{pair % 80}", "img2": f"i{(3 * pair + 7) % 80}", "type_idx": types[t], "cluster1": pair % 5,
                "outcomes": [(types[t], bool((pair + t) % 2))]}
    cands = [make(p, t, False) for p in range(n_pairs) for t in range((p % 4) + 1)]
    labeled = [make(100 + p, t, True) for p in range(labeled_pairs) for t in range(2)]
    np.random.default_rng(0).shuffle(cands); return cands, labeled, cache


SYNTHETIC = sorted(set(ju.ROW_RULES) | set(ju.PAIR_RULES))


@pytest.mark.parametrize("name", SYNTHETIC)
def test_selector_returns_budget_distinct_judgments_reproducibly(name):
    cands, labeled, cache = _judgments(); select = ju.make_selector(name, None)
    first = [x["pair_id"] for x in select(cands, labeled, _Model(), cache, 12, 3)]; second = [x["pair_id"] for x in select(cands, labeled, _Model(), cache, 12, 3)]
    assert first == second and len(first) == 12 and len(set(first)) == 12 and set(first) <= {x["pair_id"] for x in cands}


@pytest.mark.parametrize("name", sorted(ju.PAIR_RULES))
def test_pair_level_rules_reveal_one_judgment_per_pair(name):
    cands, labeled, cache = _judgments(); chosen = ju.make_selector(name, None)(cands, labeled, _Model(), cache, 12, 3)
    assert len({x["base_pair"] for x in chosen}) == 12


def test_pair_level_rule_falls_back_to_a_second_pass_when_budget_exceeds_pairs():
    cands, labeled, cache = _judgments(n_pairs=8); cands = [c for c in cands if c["base_pair"] in {f"p{i}" for i in range(8)}]
    chosen = ju.make_selector("core_set_relation", None)(cands, labeled, _Model(), cache, min(len(cands), 14), 1)
    assert len({x["pair_id"] for x in chosen}) == len(chosen) == min(len(cands), 14) > 8


def test_row_rules_can_choose_the_type():
    cands, labeled, cache = _judgments()
    chosen = ju.make_selector("laplace_bald", None)(cands, labeled, _Model(), cache, 20, 0)
    assert len({x["type_idx"] for x in chosen}) > 1


def test_own_head_uncertainty_uses_the_candidates_type():
    cands, labeled, cache = _judgments(); model = _Model(); u = ju._own_uncertainty(model, cands, cache)
    _, p = __import__("new_pair_strategies")._hidden_and_logit(model, cands, cache)
    own = p[np.arange(len(cands)), [c["type_idx"] for c in cands]]; expected = -(own * np.log(own) + (1 - own) * np.log(1 - own)) / np.log(2)
    assert np.allclose(u, expected, atol=1e-5)


# ------------------------------------------------------------------ training uses exactly the selected judgment rows
class _Features:
    def __init__(self):
        torch.manual_seed(0); self.base_model = nn.Module(); self.base_model.reward_head = nn.Sequential(nn.Linear(512, 8), nn.ReLU(), nn.Dropout(.2), nn.Linear(8, 5))
        self.base_model.eval(); self.seen = []
    def get(self, paths):
        paths = list(paths); self.seen.append(paths)
        return torch.stack([torch.full((512,), float(int(str(p)[1:]))) for p in paths]) if paths else torch.empty(0, 512)


class _Exp:
    class cfg: weight_decay = 1e-4; bad_anchor_weight = .1
    references = {}; bad_paths = []
    def __init__(self):
        rows = []
        for pair in range(3):
            for t in (0, 2, 3, 4):  # one pair has four judgments with distinct images per type so that a leaked row is detectable
                rows.append({"pair_id": f"p{pair}", "resolved_img1": f"i{100 * pair + t}", "resolved_img2": f"i{100 * pair + t + 10}", "type_idx": t, "Winner": "1", "confidence_weight": 1.0})
        frame = pd.DataFrame(rows); self.groups = {k: g.reset_index(drop=True) for k, g in frame.groupby("pair_id")}
    def rows_for(self, ids): return pd.concat([self.groups[x] for x in ids], ignore_index=True) if ids else pd.DataFrame()


def test_unselected_judgment_of_a_selected_pair_is_not_trained_on(monkeypatch):
    import judgment_unit_study as study
    exp, features = _Exp(), _Features(); captured = {}
    monkeypatch.setattr(frozen, "fit_head", lambda head, xa, xb, typ, weight, winner, refs, bad, lr, steps, *a, **k: captured.update(xa=xa, xb=xb, typ=typ) or head)
    chosen = [("p0", 1), ("p2", 3)]  # one judgment each; the other judgments of p0 and p2 stay hidden
    frozen.train_model(exp, features, [], .01, 3, rows=study.rows_of(exp, chosen))
    assert len(captured["typ"]) == 2
    expected = {(float(int(exp.groups[p].iloc[pos].resolved_img1[1:])), float(int(exp.groups[p].iloc[pos].resolved_img2[1:]))) for p, pos in chosen}
    assert {(float(a[0]), float(b[0])) for a, b in zip(captured["xa"], captured["xb"])} == expected
    assert sorted(captured["typ"].tolist()) == sorted(int(exp.groups[p].iloc[pos].type_idx) for p, pos in chosen)


def test_group_path_still_trains_on_all_rows_of_the_group(monkeypatch):
    exp, features = _Exp(), _Features(); captured = {}
    monkeypatch.setattr(frozen, "fit_head", lambda head, xa, xb, typ, weight, winner, refs, bad, lr, steps, *a, **k: captured.update(typ=typ) or head)
    frozen.train_model(exp, features, ["p2", "p0"], .01, 3)
    assert len(captured["typ"]) == 8


def test_rows_of_is_order_independent_and_deduplicated():
    import judgment_unit_study as study
    exp = _Exp(); a = study.rows_of(exp, [("p1", 2), ("p0", 0), ("p1", 2)]); b = study.rows_of(exp, [("p0", 0), ("p1", 2)])
    assert a.equals(b) and len(a) == 2


# ------------------------------------------------------------------ optional per-budget head schedule (re-tuned run); default behaviour unchanged
def _schedule_table():
    return {"initial": {"learning_rate": .001, "steps": 100}, "10": {"learning_rate": .003, "steps": 300}, "20": {"learning_rate": .01, "steps": 100},
            "40": {"learning_rate": .001, "steps": 1000}, "60": {"learning_rate": .003, "steps": 100}}


@pytest.mark.parametrize("n,expected", [(0, (.001, 100)), (10, (.003, 300)), (20, (.01, 100)), (30, (.001, 1000)), (40, (.001, 1000)), (50, (.003, 100)), (60, (.003, 100)), (70, (.003, 100))])
def test_schedule_for_uses_the_next_calibrated_budget(n, expected):
    import judgment_unit_study as study
    assert study.schedule_for(_schedule_table(), n) == expected


def test_load_schedule_ignores_documentation_keys(tmp_path):
    import json, judgment_unit_study as study
    raw = {**_schedule_table(), "selection": "text", "pooled_over_budgets_for_reference": {"learning_rate": .01, "steps": 100}}; path = tmp_path / "s.json"; path.write_text(json.dumps(raw))
    assert study.load_schedule(path) == _schedule_table()


def _bare_context(study, schedule):
    ctx = study.Context.__new__(study.Context); ctx.exp = _Exp(); ctx.features = _Features(); ctx.initial = [("p0", 0), ("p0", 1)]; ctx.lr, ctx.steps, ctx.schedule = .01, 100, schedule
    return ctx


def test_context_fit_default_ignores_the_schedule_machinery(monkeypatch):
    import judgment_unit_study as study
    seen = []; monkeypatch.setattr(frozen, "train_model", lambda exp, features, ids, lr, steps, **k: seen.append((lr, steps)) or None)
    ctx = _bare_context(study, None)
    for extra in ([], [("p1", 0)], [("p1", 0), ("p1", 1), ("p2", 0)]): ctx.fit(ctx.initial + extra)
    assert seen == [(.01, 100)] * 3


def test_context_fit_with_schedule_uses_the_entry_for_the_number_of_revealed_judgments(monkeypatch):
    import judgment_unit_study as study
    seen = []; monkeypatch.setattr(frozen, "train_model", lambda exp, features, ids, lr, steps, **k: seen.append((lr, steps)) or None)
    ctx = _bare_context(study, _schedule_table()); extra_pool = [(f"p{p}", t) for p in (1, 2) for t in range(4)]
    for n in (0, 3, 8): ctx.fit(ctx.initial + extra_pool[:n])   # initial; 3 revealed -> budget 10 entry; 8 revealed -> budget 10 entry
    assert seen == [(.001, 100), (.003, 300), (.003, 300)]
    assert ctx.params_for(ctx.initial + extra_pool[:3] + [("p0", 0)]) == (.003, 300)   # an initial judgment listed twice is not counted as revealed


def test_calibration_selection_is_per_budget_with_the_old_tie_break():
    import judgment_unit_calibrate as cal
    rows = []
    for budget, best in ((0, (.001, 100)), (10, (.01, 300))):
        for lr, steps in cal.GRID:
            for seed in (1, 2): rows.append({"seed": seed, "budget": budget, "learning_rate": lr, "steps": steps, "decisive_log_loss": .3 if (lr, steps) == best else .5, "decisive_accuracy": .8})
    rows += [{"seed": s, "budget": 10, "learning_rate": .003, "steps": 1000, "decisive_log_loss": .3, "decisive_accuracy": .8} for s in (1, 2)]   # tie with (.01, 300): fewer steps wins
    schedule, _ = cal.choose(pd.DataFrame(rows))
    assert schedule == {"initial": {"learning_rate": .001, "steps": 100}, "10": {"learning_rate": .01, "steps": 300}}


# ------------------------------------------------------------------ integration on the real data
DATA = STUDY.parents[2] / "data" / "original data"


@pytest.mark.skipif(not DATA.exists() or not (STUDY / "results" / "frozen_encoder_task3" / "simclr_feature_cache.pt").exists(), reason="needs the data and the feature cache")
@pytest.mark.parametrize("split", ["A", "B"])
def test_every_strategy_obeys_the_judgment_unit_invariants_on_real_data(split, tmp_path):
    import judgment_unit_study as study
    import run_pair_endpoint_study as single
    torch.set_num_threads(2); cache = frozen.load_feature_cache(single.CACHE, single.DATA)
    ctx = study.Context(42, tmp_path, cache, split, "groups", .01, 20)
    assert len(ctx.pool) == sum(len(ctx.exp.groups[g]) for g in ctx.pool_groups) and len(ctx.initial) == sum(len(ctx.exp.groups[g]) for g in ctx.initial_groups)
    assert not (set(ctx.initial) & set(ctx.pool))
    model = ctx.fit(ctx.initial); state = {"cands": study.make_candidates(ctx, ctx.pool, ctx.initial, model, 42)}
    for name in [n for n in ju.STRATEGY_FAMILY if n != "random"]:
        selector = None if name in ju.ENSEMBLE else ju.make_selector(name, ctx.exp)
        picks = study.choose(ctx, name, selector, ctx.pool, ctx.initial, model, 10, 42, state)
        assert len(set(picks)) == 10 and set(picks) <= set(ctx.pool), name
        rows = study.rows_of(ctx.exp, ctx.initial + picks); assert len(rows) == len(ctx.initial) + 10, name   # no sibling judgment comes along
    random_initial = study.Context(42, tmp_path, cache, split, "random", .01, 20)
    assert len(random_initial.initial) == 10 and not (set(random_initial.initial) & set(random_initial.pool))

"""New selectors return distinct, valid candidates and respect the budget on synthetic features."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"))
from new_pair_strategies import NEW_STRATEGIES  # noqa: E402


class _Model:
    def __init__(self):
        torch.manual_seed(0); self.reward_head = nn.Sequential(nn.Linear(512, 16), nn.ReLU(), nn.Dropout(.2), nn.Linear(16, 5))


def _data(n=40, labeled=8):
    g = torch.Generator().manual_seed(1); cache = {f"i{k}": torch.randn(512, generator=g) for k in range(60)}
    def item(k): return {"pair_id": f"p{k}", "img1": f"i{k % 60}", "img2": f"i{(3 * k + 7) % 60}", "type_idx": int([0, 2, 3, 4][k % 4])}
    return [item(k) for k in range(n)], [item(100 + k) for k in range(labeled)], cache


@pytest.mark.parametrize("name", sorted(NEW_STRATEGIES))
def test_selector_returns_budget_distinct_candidates(name):
    candidates, labeled, cache = _data(); chosen = NEW_STRATEGIES[name](candidates, labeled, _Model(), cache, budget=12, seed=3)
    ids = [x["pair_id"] for x in chosen]
    assert len(ids) == 12 and len(set(ids)) == 12 and set(ids) <= {x["pair_id"] for x in candidates}


@pytest.mark.parametrize("name", sorted(NEW_STRATEGIES))
def test_selector_is_deterministic_for_a_seed(name):
    candidates, labeled, cache = _data()
    first = [x["pair_id"] for x in NEW_STRATEGIES[name](candidates, labeled, _Model(), cache, budget=10, seed=5)]
    second = [x["pair_id"] for x in NEW_STRATEGIES[name](candidates, labeled, _Model(), cache, budget=10, seed=5)]
    assert first == second


@pytest.mark.parametrize("name", sorted(NEW_STRATEGIES))
def test_selector_does_not_read_the_label_derived_type(name):
    candidates, labeled, cache = _data()
    stripped = [{k: v for k, v in x.items() if k != "type_idx"} for x in candidates]; stripped_labeled = [{k: v for k, v in x.items() if k != "type_idx"} for x in labeled]
    a = [x["pair_id"] for x in NEW_STRATEGIES[name](candidates, labeled, _Model(), cache, budget=10, seed=2)]
    b = [x["pair_id"] for x in NEW_STRATEGIES[name](stripped, stripped_labeled, _Model(), cache, budget=10, seed=2)]
    assert a == b


def test_bald_decisive_uses_labelled_outcomes_when_available():
    candidates, labeled, cache = _data(); rng = np.random.default_rng(0)
    with_outcomes = [{**x, "outcomes": [(int(k), bool(rng.random() > .5)) for k in (0, 2, 3, 4)]} for x in labeled]
    ids = [x["pair_id"] for x in NEW_STRATEGIES["bald_decisive"](candidates, with_outcomes, _Model(), cache, budget=10, seed=1)]
    assert len(ids) == 10 and len(set(ids)) == 10

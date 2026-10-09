"""New acquisition rules (new_methods_strategies): budget distinct candidates, reproducible, and no use of the hidden outcome of a candidate."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

STUDY = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import new_methods_strategies as nm  # noqa: E402
from test_judgment_unit import _Model, _judgments  # noqa: E402


class _Ctx:
    seed = 0
    def fit(self, judgments, head_seed=None): return _Model()


@pytest.fixture(autouse=True)
def _ctx():
    nm.CTX.clear(); nm.CTX["ctx"] = _Ctx(); yield; nm.CTX.clear()


@pytest.mark.parametrize("name", sorted(nm.NEW))
def test_budget_distinct_reproducible_and_outcome_blind(name):
    cands, labeled, cache = _judgments(); model = _Model()
    a = nm.NEW[name](cands, labeled, model, cache, 12, 0); b = nm.NEW[name](cands, labeled, model, cache, 12, 0)
    assert len(a) == 12 and len({x["pair_id"] for x in a}) == 12 and {x["pair_id"] for x in a} <= {x["pair_id"] for x in cands}
    assert [x["pair_id"] for x in a] == [x["pair_id"] for x in b]
    flipped = copy.deepcopy(cands)
    for x in flipped: x["outcomes"] = [(t, not ok) for t, ok in x["outcomes"]]   # hidden outcomes of candidates must not matter
    c = nm.NEW[name](flipped, labeled, model, cache, 12, 0)
    assert [x["pair_id"] for x in a] == [x["pair_id"] for x in c]


def test_budget_larger_than_pool_returns_pool():
    cands, labeled, cache = _judgments(n_pairs=4); model = _Model()
    assert len(nm.NEW["vopt"](cands, labeled, model, cache, 10 ** 3, 0)) == len(cands)

"""Frozen-encoder head training must depend on the set of labelled rows, not their order."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"))
from frozen_encoder_reward_head import fit_head  # noqa: E402


def _problem(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    xa, xb = torch.randn(n, 512, generator=g), torch.randn(n, 512, generator=g)
    typ = torch.randint(0, 5, (n,), generator=g); weight = torch.where(torch.rand(n, generator=g) > .5, 1.0, .7)
    winner = np.array((["1", "2", "tie", "not_apply"] * n)[:n])
    refs = {"(1 x 1)": torch.randn(4, 512, generator=g), "HTR": torch.randn(4, 512, generator=g)}
    return xa, xb, typ, weight, winner, refs


def _head(seed=1):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(512, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 5))


def test_row_order_does_not_change_trained_head():
    xa, xb, typ, weight, winner, refs = _problem(); perm = torch.randperm(len(typ), generator=torch.Generator().manual_seed(3))
    first = fit_head(_head(), xa, xb, typ, weight, winner, refs, None, lr=3e-3, steps=60)
    second = fit_head(_head(), xa[perm], xb[perm], typ[perm], weight[perm], winner[perm.numpy()], refs, None, lr=3e-3, steps=60)
    for p, q in zip(first.parameters(), second.parameters()):
        assert torch.allclose(p, q, atol=1e-4), float((p - q).abs().max())


def test_repeat_training_is_identical_and_data_changes_the_head():
    xa, xb, typ, weight, winner, refs = _problem()
    a = fit_head(_head(), xa, xb, typ, weight, winner, refs, None, lr=3e-3, steps=30)
    b = fit_head(_head(), xa, xb, typ, weight, winner, refs, None, lr=3e-3, steps=30)
    c = fit_head(_head(), xa[:30], xb[:30], typ[:30], weight[:30], winner[:30], refs, None, lr=3e-3, steps=30)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert any(not torch.allclose(p, q, atol=1e-4) for p, q in zip(a.parameters(), c.parameters()))

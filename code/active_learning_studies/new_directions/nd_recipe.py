"""Recipe search around ``lap3_k10_st4`` (study ``claude/recipe-20261013``): one flexible fit routine, every knob defaults to the incumbent.

``fit_recipe`` is ``nd_graph_learner.fit_mlp_graph`` (aw8_all + Laplacian term, per-budget head schedule, all ideal images as anchors) with extra switches; with the default ``Spec`` it reproduces
``lap3_k10_st4`` to the last digit (unit test ``test_recipe.py``).  Switches (each default = incumbent):
  hidden, depth            head width / number of hidden layers
  dropout                  dropout probability, ACTIVE during training (the repository trains its head in eval mode, so the head's built-in Dropout(0.2) is inert by default); 0 = incumbent
  wd                       AdamW weight decay (None = the repository's value)
  tie                      'l1' (|a-b|, incumbent) | 'rk' (Rao-Kupper tie likelihood with a learned margin) | 'soft' (soft 0.5 target of the Bradley-Terry likelihood)
  tie_w, na_w              multipliers of the weight of tie rows / not_apply rows
  mix, mix_alpha           pair-level mixup coefficient (extra loss on convex combinations of two decisive rows of the same type; soft target) and the Beta(alpha, alpha) parameter
  cw_pow                   confidence weight ** cw_pow (renormalised to the same mean weight)
  aw, anchor_form          reference-anchor weight and 'logistic' (incumbent) | 'hinge' (margin 1)
  type_lr                  per-type multiplier of the update of the final-layer row of each type
  betas, cosine, clip      AdamW betas, cosine learning-rate decay to 0, gradient-norm clip (None = off)
  full_graph               Laplacian term over ALL edges (all nodes scored every step) instead of 256 sampled edges
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nd_core as core   # first: puts the study directories on sys.path
import judgment_unit_study as ju_study
import nd_graph_learner as G

ACTIVE = G.ACTIVE


@dataclass(frozen=True)
class Spec:
    k: int = 10; lap: float = 3.0; steps_scale: float = 4.0; lr_scale: float = 1.0; aw: float = 8.0
    hidden: int = 256; depth: int = 1; dropout: float = 0.0; wd: float | None = None
    tie: str = "l1"; tie_w: float = 1.0; na_w: float = 1.0
    mix: float = 0.0; mix_alpha: float = 0.4; cw_pow: float = 1.0
    anchor_form: str = "logistic"; type_lr: tuple = (); betas: tuple = (0.9, 0.999); cosine: bool = False; clip: float | None = 1.0; full_graph: bool = False
    shuffle: bool = False
    distill: float = 0.0     # weight of the pair-difference distillation of a teacher's node scores (needs ``teacher`` in fit_recipe); 0 = off


def make_head(dim: int, seed: int, hidden: int, depth: int, dropout: float) -> nn.Module:
    """depth = 1, dropout = 0.2 is ``nd_core.new_head``; extra hidden layers keep the same width."""
    torch.manual_seed(seed); layers: list = [nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
    for _ in range(depth - 1): layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
    return nn.Sequential(*layers, nn.Linear(hidden, 5))


def _anchor_loss(scores: dict, cols: dict, form: str) -> torch.Tensor:
    values = []
    for p, sp in scores.items():
        for o, so in scores.items():
            if o != p:
                diff = sp[:, cols[p]][:, None] - so[:, cols[p]][None, :]
                values.append((-F.logsigmoid(diff) if form == "logistic" else F.relu(1.0 - diff)).mean())
    return torch.stack(values).mean()


def fit_recipe(ctx, labelled, seed: int, gd, featfn, node_x: torch.Tensor, dim: int, spec: Spec, teacher: torch.Tensor | None = None):
    lr, steps = ctx.params_for(labelled); lr, steps = lr * spec.lr_scale, int(round(steps * spec.steps_scale)); rows = ju_study.rows_of(ctx.exp, labelled)
    head = make_head(dim, seed, spec.hidden, spec.depth, spec.dropout); refs, bad = core.anchors(ctx, featfn, True)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    if spec.cw_pow != 1.0 and len(rows):
        wp = weight ** spec.cw_pow; weight = wp * (weight.mean() / wp.mean())
    masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}; index = torch.arange(len(typ))
    train_mode = spec.dropout > 0; head.train(train_mode)
    wd = ctx.exp.cfg.weight_decay if spec.wd is None else spec.wd
    margin = nn.Parameter(torch.tensor(0.5)) if spec.tie == "rk" else None      # learned Rao-Kupper margin delta = softplus(margin) >= 0
    params = list(head.parameters()) + ([margin] if margin is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=spec.betas)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / steps))) if spec.cosine else None
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    gen2 = torch.Generator().manual_seed(seed * 1000033 + 5) if (teacher is not None and spec.distill > 0) else None
    tw = torch.ones(len(ACTIVE)); last = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
    row_scale = None
    if spec.type_lr:
        row_scale = torch.ones(5)
        for t, m in spec.type_lr: row_scale[int(t)] = float(m)
    decisive = torch.as_tensor(np.isin(winner, ["1", "2"])) if len(rows) else torch.empty(0, dtype=torch.bool)
    ysoft = torch.as_tensor((winner == "1").astype(np.float32)) if len(rows) else torch.empty(0)
    mix_ok = None; beta = torch.distributions.Beta(spec.mix_alpha, spec.mix_alpha)
    if spec.mix > 0 and len(rows) > 1:
        # candidate partners: decisive rows of the same type
        mix_ok = decisive[:, None] & decisive[None, :] & (typ[:, None] == typ[None, :]) & ~torch.eye(len(typ), dtype=torch.bool)
        if not mix_ok.any(): mix_ok = None
    for _ in range(steps):
        a, b = head(xa)[index, typ], head(xb)[index, typ]; d = a - b
        if spec.tie == "rk":
            delta = F.softplus(margin); p_tie = (1 - torch.sigmoid(d - delta) - torch.sigmoid(-d - delta)).clamp_min(1e-6)
            tie_term = -torch.log(p_tie); win1 = -F.logsigmoid(d - delta); win2 = -F.logsigmoid(-d - delta)
        else:
            win1, win2 = -F.logsigmoid(d), -F.logsigmoid(-d)
            tie_term = d.abs() if spec.tie == "l1" else -0.5 * (F.logsigmoid(d) + F.logsigmoid(-d))
        terms = win1 * masks["1"] + win2 * masks["2"] + tie_term * masks["tie"] * spec.tie_w + (F.relu(a) + F.relu(b)) * masks["not_apply"] * spec.na_w
        loss = (terms * weight).sum() / max(len(typ), 1)
        if mix_ok is not None:   # pair-level mixup: convex combination of two decisive rows of the same type (features and soft target); rows without a partner are masked out
            n = len(typ); has = mix_ok.any(1); probs = torch.where(has[:, None], mix_ok.float(), torch.ones(n, n)); partner = torch.multinomial(probs, 1, generator=gen)[:, 0]
            lam = beta.sample((n,)); l = lam[:, None]; ok = (has & mix_ok[index, partner]).float()
            dm = head(l * xa + (1 - l) * xa[partner])[index, typ] - head(l * xb + (1 - l) * xb[partner])[index, typ]
            t = lam * ysoft + (1 - lam) * ysoft[partner]; w = lam * weight + (1 - lam) * weight[partner]
            loss = loss + spec.mix * (-(t * F.logsigmoid(dm) + (1 - t) * F.logsigmoid(-dm)) * w * ok).sum() / max(n, 1)
        if len(refs) > 1: loss = loss + spec.aw * _anchor_loss({nm: head(f) for nm, f in refs.items()}, cols, spec.anchor_form)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if spec.lap > 0 and gd is not None:
            if spec.full_graph:
                s = head(node_x)[:, active]; si, sj = s[gd.edges[:, 0]], s[gd.edges[:, 1]]
            else:
                e = gd.edges[torch.randint(gd.n_edges, (G.EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]
            loss = loss + spec.lap * (((si - sj) ** 2) * tw).sum(1).mean()
        if gen2 is not None:   # distillation of a teacher's (offset-free) pair differences of node scores: random node pairs + graph edges
            gi, gj = torch.randint(len(teacher), (G.EDGE_BATCH,), generator=gen2), torch.randint(len(teacher), (G.EDGE_BATCH,), generator=gen2); ee = gd.edges[torch.randint(gd.n_edges, (G.EDGE_BATCH,), generator=gen2)]
            ii, jj = torch.cat([gi, ee[:, 0]]), torch.cat([gj, ee[:, 1]]); si, sj = head(node_x[ii])[:, active], head(node_x[jj])[:, active]
            loss = loss + spec.distill * (((si - sj) - (teacher[ii] - teacher[jj])) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward()
        if spec.clip is not None: torch.nn.utils.clip_grad_norm_(params, spec.clip)
        if row_scale is not None: before = [last.weight.detach().clone(), last.bias.detach().clone()]
        opt.step()
        if row_scale is not None:
            with torch.no_grad(): last.weight.copy_(before[0] + row_scale[:, None] * (last.weight - before[0])); last.bias.copy_(before[1] + row_scale * (last.bias - before[1]))
        if sched is not None: sched.step()
    return head.eval()


def recipe_learner(spec: Spec):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, spec.k, spec.shuffle, 0.0); featfn = core.identity_features(ctx); node_x = gd.x; dim = gd.x.shape[1]
        head = fit_recipe(ctx, labelled, seed, gd, featfn, node_x, dim, spec)
        return lambda test: core.head_d(head, featfn, test)
    return learner


BASE = Spec()
# development variants (one-at-a-time around the incumbent) -- the registry is the complete list of what was tried
VARIANTS: dict[str, Spec] = {
    "rc_fullg": replace(BASE, full_graph=True),
    "rc_aw4": replace(BASE, aw=4.0), "rc_aw16": replace(BASE, aw=16.0),
    "rc_hinge": replace(BASE, anchor_form="hinge"),
    "rc_w512": replace(BASE, hidden=512), "rc_w1024": replace(BASE, hidden=1024), "rc_w128": replace(BASE, hidden=128),
    "rc_d2": replace(BASE, depth=2), "rc_d3": replace(BASE, depth=3),
    "rc_do0.3": replace(BASE, dropout=0.3), "rc_wd1": replace(BASE, wd=1.0),
    "rc_rk": replace(BASE, tie="rk"), "rc_softtie": replace(BASE, tie="soft"),
    "rc_dn0.3": replace(BASE, tie_w=0.3, na_w=0.3),
    "rc_mix": replace(BASE, mix=1.0),
    # confidence-weight power is NOT screened: ≥ 95 % of the judgments have confidence_weight exactly 1.0 and the rest 0.7 (tested in test_recipe.py), so the knob is almost a no-op
    "rc_ptlr": replace(BASE, type_lr=((3, 2.0), (4, 2.0))),
    "rc_b99": replace(BASE, betas=(0.9, 0.99)), "rc_cos": replace(BASE, cosine=True), "rc_noclip": replace(BASE, clip=None),
    "rc_base": BASE,
}
LEARNERS = {name: recipe_learner(spec) for name, spec in VARIANTS.items()}


def combine(*names: str) -> Spec:
    """Spec that applies every switch of the named single variants (``rc_x``) at once; two variants that change the same field are a conflict."""
    from dataclasses import fields
    out = {}
    for n in names:
        for f in fields(Spec):
            v, b = getattr(VARIANTS[n], f.name), getattr(BASE, f.name)
            if v != b:
                assert f.name not in out, f"conflicting switches for {f.name}: {names}"; out[f.name] = v
    return replace(BASE, **out)


def resolve(name: str):
    """Registry lookup; ``rcx_<a>+<b>`` (e.g. ``rcx_rc_d2+rc_rk``) is the combination of two or more single variants."""
    if name in LEARNERS: return LEARNERS[name]
    if name.startswith("rcx_"): return recipe_learner(combine(*name[4:].split("+")))
    raise KeyError(name)

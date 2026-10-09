#!/usr/bin/env python3
"""Judgment-unit study: the query unit is ONE (pair, reconstruction type) judgment and the budget counts judgments.

Same splits, seeds, features, head schedule, held-out sets and endpoint as ``run_pair_endpoint_study.py`` / ``run_pair_endpoint_sequential.py`` (they are imported,
not copied).  What changes: the candidate pool is every judgment row of the pool groups; revealing a candidate reveals that single row (its own winner label)
and nothing else; the budget (10, 20, 40, 60) counts revealed judgments.

Environment variables
  JU_SPLIT    A (default; 10 initial / 20 validation / 40 test groups) or B (classifier2-style 20% hold-out, no validation)
  JU_INITIAL  groups (default): the initial 10 groups' judgments are revealed, as in the earlier study (cold-start data);
              random: 10 random judgments drawn from the judgments of the initial and pool groups (sensitivity check)
  JU_MODE     single | sequential | both (default both)
  PAIR_STUDY_SEEDS, JU_ONLY (comma list of strategies), JU_OUT (output sub-folder, default <split>_<initial>)
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
import judgment_unit_strategies as ju
import pair_preference_endpoint as endpoint
import run_pair_endpoint_study as single_study

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "judgment_unit_study"
BUDGETS = (10, 20, 40, 60); ROUND = 10; INITIAL_RANDOM_JUDGMENTS = 10
ENSEMBLE_SIZE = 8


def strategy_names() -> list[str]:
    reps = lambda base: [base] + [f"{base}_r{r}" for r in range(1, ju.RANDOM_REPLICATES)]
    non_random = [n for n in ju.STRATEGY_FAMILY if n != "random"]
    names = reps("random") + non_random + reps(ju.EXTRA_BASELINE)
    if os.environ.get("JU_GRAPH"):  # graph-aware strategies are opt-in; JU_ONLY then restricts the run (e.g. to the graph names only)
        import graph_strategies, graph_typed; names += list(graph_strategies.GRAPH_NAMES) + list(graph_typed.TYPED_NAMES)
    return names


def family_of(name: str) -> str:
    return name.rsplit("_r", 1)[0] if name.startswith("random") and name[-1].isdigit() and "_r" in name else name


# ------------------------------------------------------------------------------------------------ judgments
def judgments_of(exp, group_ids) -> list[tuple[str, int]]:
    return [(pid, pos) for pid in sorted(group_ids) for pos in range(len(exp.groups[pid]))]


def rows_of(exp, judgments) -> pd.DataFrame:
    """The training rows of exactly these judgments (order irrelevant: sorted).  No other row of the same pair is included."""
    unique = sorted(set(judgments))
    return pd.concat([exp.groups[pid].iloc[[pos]] for pid, pos in unique], ignore_index=True) if unique else pd.DataFrame()


class Context:
    """Everything one seed needs: experiment, features, splits and the judgment table."""
    def __init__(self, seed, scratch, cache, split, initial_mode, lr, steps):
        os.environ["PAIR_STUDY_SPLIT"] = "classifier2" if split == "B" else ""
        self.seed, self.lr, self.steps = seed, lr, steps
        self.exp, self.features, initial, pool, self.validation, self.test = single_study.setup(seed, scratch, cache)
        self.initial_groups, self.pool_groups = list(initial), list(pool)
        if initial_mode == "groups": self.initial = judgments_of(self.exp, initial); self.pool = judgments_of(self.exp, pool)
        elif initial_mode == "random":
            union = judgments_of(self.exp, list(initial) + list(pool)); chosen = set(random.Random(seed * 1000 + 53).sample(union, INITIAL_RANDOM_JUDGMENTS))
            self.initial = sorted(chosen); self.pool = [j for j in union if j not in chosen]
        else: raise ValueError(initial_mode)
        self.info = {}
        for group in set(initial) | set(pool):
            for pos, r in enumerate(self.exp.groups[group].itertuples()):
                self.info[(group, pos)] = {"img1": r.resolved_img1, "img2": r.resolved_img2, "type_idx": int(r.type_idx), "decisive": r.Winner in ("1", "2"), "winner": r.Winner}
        held_out = {x for g in list(self.validation) + list(self.test) for r in self.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
        used = {x for j in self.initial + self.pool for x in (self.info[j]["img1"], self.info[j]["img2"])}
        assert not (held_out & used), "validation/test image reachable from the labelled set or the candidate pool"
        assert not (set(self.initial) & set(self.pool)); self.pool_index = {j: i for i, j in enumerate(self.pool)}

    def fit(self, judgments, head_seed=None):
        return frozen.train_model(self.exp, self.features, [], self.lr, self.steps, head_seed=head_seed, rows=rows_of(self.exp, judgments))

    def item(self, j, cluster=None) -> dict:
        d = self.info[j]; out = {"pair_id": f"{j[0]}#{j[1]}", "base_pair": j[0], "pos": j[1], "img1": d["img1"], "img2": d["img2"], "type_idx": d["type_idx"],
                                 "outcomes": [(d["type_idx"], d["decisive"])]}
        if cluster is not None: out["cluster1"] = cluster
        return out

    def evaluate(self, model) -> dict:
        return endpoint.evaluate_full(self.exp, self.features, model, self.validation, self.test)

    def manifest(self) -> dict:
        return {"seed": self.seed, "initial": self.initial, "pool": [[p, pos, self.info[(p, pos)]["type_idx"], self.info[(p, pos)]["winner"]] for p, pos in self.pool],
                "validation_groups": self.validation, "test_groups": self.test, "initial_groups": self.initial_groups, "pool_groups": self.pool_groups}


# ------------------------------------------------------------------------------------------------ selection
def make_candidates(ctx: Context, remaining, labeled, model, order_seed):
    """Candidate dicts for the remaining judgments (clusters from the unique pairs), shuffled with ``order_seed``; labelled dicts; embedding cache."""
    pair_ids = sorted({p for p, _ in remaining}); pairs, _ = ctx.exp.candidates_with_clusters(pair_ids, model); cluster = {x["pair_id"]: x["cluster1"] for x in pairs}
    order = list(remaining); random.Random(order_seed * 7919 + 3).shuffle(order)
    cands = [ctx.item(j, cluster[j[0]]) for j in order]; labeled_items = [ctx.item(j) for j in labeled]
    reference_items = [{"img1": str(p), "img2": str(p)} for paths in ctx.exp.references.values() for p in paths]  # typed reference (ideal) images: training anchors of every strategy
    return cands, labeled_items, ctx.features.embedding_cache(cands + labeled_items + reference_items)


def random_picks(ctx, name, remaining, budget, key):
    rng = random.Random(key); remaining = list(remaining)
    if not name.startswith(ju.EXTRA_BASELINE): return rng.sample(remaining, budget)
    picks = []
    for _ in range(budget):  # uniform pair among pairs with a remaining judgment, then uniform type of it
        pairs = sorted({p for p, _ in remaining}); pair = rng.choice(pairs); choice = rng.choice([j for j in remaining if j[0] == pair]); picks.append(choice); remaining.remove(choice)
    return picks


def choose(ctx, name, selector, remaining, labeled, model, budget, seed, state):
    """One acquisition batch of ``budget`` judgments from ``remaining``."""
    if name in ju.ENSEMBLE:
        if "ensemble" not in state: state["ensemble"] = [ctx.fit(labeled, head_seed=1000 + k) for k in range(ENSEMBLE_SIZE)]
        cands, labeled_items, cache = state["cands"]
        picked = ju.ENSEMBLE[name](cands, labeled_items, state["ensemble"], cache, budget, seed)
    else:
        cands, labeled_items, cache = state["cands"]; picked = selector(cands, labeled_items, model, cache, budget, seed)
    ids = {f"{j[0]}#{j[1]}": j for j in remaining}; result = [ids[x["pair_id"]] for x in picked]
    assert len(result) == budget and len(set(result)) == budget and set(result) <= set(remaining), (name, len(result), budget)
    return result


def metrics_for(ctx, model, labeled_after, budget) -> dict:
    revealed = [j for j in labeled_after if j not in set(ctx.initial)]
    assert len(revealed) == budget, (len(revealed), budget)
    return {**ctx.evaluate(model), "n_revealed": len(revealed), "n_pairs_touched": len({p for p, _ in revealed}), "n_labeled_total": len(labeled_after),
            "selected": sorted(ctx.pool_index[j] for j in revealed)}


def run_single(ctx: Context, names, cells: Path) -> None:
    baseline = None; state = {}; selectors = {}
    for name in names:
        target = cells / f"seed{ctx.seed}_{name}.json"
        if target.exists(): continue
        if baseline is None:
            baseline = ctx.fit(ctx.initial); state["cands"] = make_candidates(ctx, ctx.pool, ctx.initial, baseline, ctx.seed)
        out = {}
        base = family_of(name); replicate = int(name.rsplit("_r", 1)[1]) if name != base else 0
        for budget in BUDGETS:
            if base in ("random", ju.EXTRA_BASELINE): picks = random_picks(ctx, name, ctx.pool, budget, ctx.seed * 7919 + budget * 31 + replicate)
            else:
                if base not in selectors and base not in ju.ENSEMBLE: selectors[base] = ju.make_selector(base, ctx.exp)
                picks = choose(ctx, base, selectors.get(base), ctx.pool, ctx.initial, baseline, budget, ctx.seed, state)
            labeled_after = ctx.initial + picks; out[str(budget)] = metrics_for(ctx, ctx.fit(labeled_after), labeled_after, budget)
        target.write_text(json.dumps({"seed": ctx.seed, "strategy": name, "condition": "single", "pool_judgments": len(ctx.pool), "initial_judgments": len(ctx.initial), "checkpoints": out}, separators=(",", ":")))


def trajectory(ctx: Context, name) -> dict:
    base = family_of(name); replicate = int(name.rsplit("_r", 1)[1]) if name != base else 0
    selector = None if base in ("random", ju.EXTRA_BASELINE) or base in ju.ENSEMBLE else ju.make_selector(base, ctx.exp)
    labeled = list(ctx.initial); result = {}; model = ctx.fit(labeled)
    for round_index in range(max(BUDGETS) // ROUND):
        remaining = [j for j in ctx.pool if j not in set(labeled)]
        if selector is None and base not in ju.ENSEMBLE:
            picks = random_picks(ctx, name, remaining, ROUND, ctx.seed * 104729 + round_index * 31 + replicate)
        else:
            state = {"cands": make_candidates(ctx, remaining, labeled, model, ctx.seed + round_index)}
            picks = choose(ctx, base, selector, remaining, labeled, model, ROUND, ctx.seed + round_index, state)
        labeled = labeled + picks; model = ctx.fit(labeled); acquired = (round_index + 1) * ROUND
        if acquired in BUDGETS: result[str(acquired)] = metrics_for(ctx, model, labeled, acquired)
    return result


def run_sequential(ctx: Context, names, cells: Path) -> None:
    for name in names:
        target = cells / f"seed{ctx.seed}_{name}.json"
        if target.exists(): continue
        target.write_text(json.dumps({"seed": ctx.seed, "strategy": name, "condition": "sequential", "round_size": ROUND, "pool_judgments": len(ctx.pool),
                                      "initial_judgments": len(ctx.initial), "checkpoints": trajectory(ctx, name)}, separators=(",", ":")))


def main() -> None:
    split = os.environ.get("JU_SPLIT", "A"); initial_mode = os.environ.get("JU_INITIAL", "groups"); mode = os.environ.get("JU_MODE", "both")
    run_dir = OUT / os.environ.get("JU_OUT", f"{split}_{initial_mode}"); torch.set_num_threads(int(os.environ.get("JU_THREADS", "2")))
    schedule = json.loads((single_study.OUT / "schedule.json").read_text()); lr, steps = schedule["learning_rate"], schedule["steps"]
    names = strategy_names()
    if os.environ.get("JU_ONLY"): only = set(os.environ["JU_ONLY"].split(",")); names = [n for n in names if family_of(n) in only or n in only]
    cache = frozen.load_feature_cache(single_study.CACHE, single_study.DATA)
    for kind in ("single", "sequential"): (run_dir / kind).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch_name:
        for seed in single_study.seeds():
            started = time.monotonic(); ctx = Context(seed, Path(scratch_name), cache, split, initial_mode, lr, steps)
            manifest = run_dir / f"manifest_seed{seed}.json"
            if not manifest.exists(): manifest.write_text(json.dumps(ctx.manifest(), separators=(",", ":")))
            cell_initial = run_dir / "single" / f"seed{seed}_initial_only.json"
            if not cell_initial.exists(): cell_initial.write_text(json.dumps({"seed": seed, "strategy": "initial_only", "pool_judgments": len(ctx.pool), "initial_judgments": len(ctx.initial), **ctx.evaluate(ctx.fit(ctx.initial))}))
            if mode in ("single", "both"): run_single(ctx, names, run_dir / "single")
            if mode in ("sequential", "both"): run_sequential(ctx, names, run_dir / "sequential")
            print(f"split {split} initial {initial_mode} seed {seed}: pool {len(ctx.pool)} judgments, {len({p for p, _ in ctx.pool})} pairs, finished in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

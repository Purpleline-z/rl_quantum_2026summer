#!/usr/bin/env python3
"""Sequential (multi-round) acquisition on the held-out pair-preference endpoint, frozen cached features, CPU.

Rounds of ``ROUND`` groups: train the head on everything labelled so far, let the strategy pick the next batch from the remaining pool, repeat.
One trajectory per (seed, strategy) is evaluated after 10, 20, 40, and 60 acquired groups.  Same split, schedule and endpoint as the single-shot study,
so the two conditions are directly comparable.
"""
from __future__ import annotations

import json
import random as pyrandom
import tempfile
import time
from pathlib import Path

import torch

import frozen_encoder_reward_head as frozen
import pair_preference_endpoint as endpoint
import run_frozen_encoder_task3 as harness
import run_pair_endpoint_study as single
from new_pair_strategies import NEW_STRATEGIES

OUT = single.OUT / "sequential_cells"; ROUND = 10; CHECKPOINTS = (10, 20, 40, 60)


def trajectory(exp, features, initial, pool, validation, test, name, seed, lr, steps) -> dict:
    labeled_ids = list(initial); remaining = list(pool); result = {}
    for round_index in range(max(CHECKPOINTS) // ROUND):
        model = frozen.train_model(exp, features, labeled_ids, lr, steps)
        rows, _ = exp.candidates_with_clusters(remaining, model)
        labeled = [{"pair_id": i, "img1": exp.groups[i].iloc[0].resolved_img1, "img2": exp.groups[i].iloc[0].resolved_img2,
                    "outcomes": [(int(r.type_idx), r.Winner in ("1", "2")) for r in exp.groups[i].itertuples()]} for i in labeled_ids]
        if name.startswith("random"): picks = [r["pair_id"] for r in pyrandom.Random(seed * 104729 + round_index * 31 + (int(name[8:]) if name != "random" else 0)).sample(rows, ROUND)]
        elif name in NEW_STRATEGIES: picks = [x["pair_id"] for x in NEW_STRATEGIES[name](rows, labeled, model, features.embedding_cache(rows + labeled), ROUND, seed + round_index)]
        else:
            exp.cfg.seed = seed + round_index  # selectors that draw randomness use the config seed; vary it per round
            picks = [x["pair_id"] for x in exp.select(name, rows, model, features.embedding_cache(rows + labeled), [], budget=ROUND, labeled_ids=labeled_ids)[0]]; exp.cfg.seed = seed
        labeled_ids += picks; remaining = [r for r in remaining if r not in set(picks)]; acquired = (round_index + 1) * ROUND
        if acquired in CHECKPOINTS:
            final = frozen.train_model(exp, features, labeled_ids, lr, steps)
            result[acquired] = {**{f"test_{k}": v for k, v in endpoint.evaluate_preferences(exp, features, final, test).items()},
                                **{f"validation_{k}": v for k, v in endpoint.evaluate_preferences(exp, features, final, validation).items()},
                                "type_accuracy": frozen.evaluate_model(exp, features, final)["test_accuracy"], "selected_pair_ids": sorted(labeled_ids[len(initial):])}
    return result


def main() -> None:
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True); cache = frozen.load_feature_cache(single.CACHE, single.DATA)
    schedule = json.loads((single.OUT / "schedule.json").read_text()); lr, steps = schedule["learning_rate"], schedule["steps"]
    names = list(single.ORIGINAL) + list(NEW_STRATEGIES) + [f"random_r{r}" for r in range(1, single.RANDOM_REPLICATES)]
    with tempfile.TemporaryDirectory() as scratch_name:
        for seed in single.seeds():
            exp, features, initial, pool, validation, test = single.setup(seed, Path(scratch_name), cache); started = time.monotonic()
            for name in names:
                target = OUT / f"seed{seed}_{name}.json"
                if target.exists(): continue
                result = trajectory(exp, features, initial, pool, validation, test, name, seed, lr, steps)
                target.write_text(json.dumps({"seed": seed, "strategy": name, "round_size": ROUND, "pool": len(pool), "checkpoints": {str(k): v for k, v in result.items()}}, indent=1))
            print(f"finished sequential seed {seed} in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

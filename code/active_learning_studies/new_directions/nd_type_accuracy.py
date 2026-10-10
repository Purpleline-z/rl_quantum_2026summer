"""Does the anchor weight change the paper's own endpoint (type accuracy of the 28 outer-test ideal images by win-rate against the references)?  Only learners that train on the references
alone are valid here (aw8_all would train on the outer-test ideal images).  Dev seeds, Split B contexts, random label sets."""
import json, sys, tempfile
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_learners as L

class M:  # minimal stand-in for BTModel: the evaluation only needs .reward_head
    def __init__(self, head): self.reward_head = head

def head_of(ctx, labelled, seed, aw):
    lr, steps = ctx.params_for(labelled); return core.fit_generic(ctx, labelled, core.identity_features(ctx), 512, lr, steps, seed, anchor_weight=aw)

def main(seeds, out):
    torch.set_num_threads(1); cache = core.feature_cache(); rows = []
    with tempfile.TemporaryDirectory() as s:
        for seed in seeds:
            ctx = core.make_ctx(seed, "B", s, cache)
            for budget in (0,) + core.BUDGETS:
                for draw in range(3 if budget else 1):
                    lab = list(ctx.initial) if budget == 0 else core.random_labelled(ctx, budget, draw)
                    for name, aw in (("baseline", 0.25), ("aw2", 2.0), ("aw8", 8.0), ("aw32", 32.0)):
                        acc = core.frozen.evaluate_model(ctx.exp, ctx.features, M(head_of(ctx, lab, seed * 7 + draw, aw)))["test_accuracy"]
                        rows.append({"seed": seed, "budget": budget, "draw": draw, "learner": name, "type_acc": acc})
            print("seed", seed, flush=True)
    Path(out).write_text(json.dumps(rows))

if __name__ == "__main__":
    a, b = (int(x) for x in sys.argv[1].split("-")); main(range(a, b + 1), sys.argv[2])

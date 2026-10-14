"""Leave-anchors-out diagnostic of the latent-type model: hide the type labels of a random 25% of the 116 non-test typed ideal images, infer their types from the remaining anchors (graph prior) and
from the outcomes of the labelled judgments (EM posteriors of the judgment images fed back as soft sources), and compare with the true types.
  python3 lt_diag.py --seeds 10700-10719 --out results/lt/diag"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, lt_model as M
import judgment_unit_study as ju_study


def seeds_of(t):
    if "-" in t: lo, hi = t.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in t.split(",")]


def true_cls(gd, i): return M.COL[gd.source_class[i]]


def diagnose(ctx, labelled, seed, frac=0.25, rho=0.15, weight=1.0, scale=1.0):
    gd = G.graph_for(ctx, 10); gs = G.graph_for(ctx, 10, True); cache = ctx.features.cache
    rng = np.random.default_rng(seed * 31 + 7); ids = sorted(gd.source_class); hidden = set(int(i) for i in rng.choice(ids, size=int(round(frac * len(ids))), replace=False))
    truth = np.array([true_cls(gd, i) for i in sorted(hidden)]); H = sorted(hidden)
    rows = ju_study.rows_of(ctx.exp, labelled); jud = M.Judgments(rows); out = {}
    near = (gd.xn[H] @ gd.xn[[gd.index[p] for p in jud.images if p in gd.index]].T).max(1).values.numpy() if len(jud.images) else np.zeros(len(H)); is_near = (1 - near) <= 0.01
    def score(q, tag):
        pred = q[H].argmax(1); p = np.clip(q[H][np.arange(len(H)), truth], 1e-3, 1); ok = (pred == truth).astype(float)
        out[f"{tag}_acc"] = float(ok.mean()); out[f"{tag}_ll"] = float(np.log(p).mean()); out[f"{tag}_acc_near"] = float(ok[is_near].mean()) if is_near.any() else float("nan"); out[f"{tag}_acc_far"] = float(ok[~is_near].mean()) if (~is_near).any() else float("nan")
    for graph, name in ((gd, ""), (gs, "shuf_")):
        src = M.source_matrix(graph, hidden=hidden); q0 = M.spread(graph, src); score(q0, name + "prior")
        prior5 = M.prior_from_q(M.node_prior(graph, jud.images, cache, q0), rho=rho)
        for mode in (("em", "prior", "random") if name == "" else ("em",)):
            belief, table, pair = M.posterior_of_images(jud, prior5, mode=mode, prior_scale=scale, rng=np.random.default_rng(seed + 5))
            extra = {graph.index[p]: belief[n] for n, p in enumerate(jud.images) if p in graph.index}; q1 = M.spread(graph, M.source_matrix(graph, extra_nodes=extra, hidden=hidden, weight=weight)); score(q1, f"{name}{mode}")
            if mode == "em" and name == "": out["table_11"] = table[1, 1].tolist(); out["table_10"] = table[1, 0].tolist(); out["table_00"] = table[0, 0].tolist()
    out["n_hidden"] = len(H); out["n_near"] = int(is_near.sum()); out["n_images"] = len(jud.images); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="10700-10719"); ap.add_argument("--splits", default="A"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--budgets", default="10,20,40,60")
    ap.add_argument("--rho", type=float, default=0.15); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--weight", type=float, default=1.0); a = ap.parse_args(); torch.set_num_threads(1)
    shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(a.seeds)) if i % n == shard]:
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); recs = []
                for budget in [int(b) for b in a.budgets.split(",")]:
                    lab = core.random_labelled(ctx, budget, 0); r = diagnose(ctx, lab, seed, rho=a.rho, weight=a.weight, scale=a.scale); r.update(seed=seed, split=split, budget=budget); recs.append(r)
                target.write_text(json.dumps(recs)); print(f"seed {seed} {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()

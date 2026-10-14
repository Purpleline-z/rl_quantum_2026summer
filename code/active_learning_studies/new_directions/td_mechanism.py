#!/usr/bin/env python3
"""Exploratory mechanism audit of the forward protocol (descriptive, not a pre-registered test): for every test judgment, how far (1 - cosine) are its images from the nearest PAST node (inductive graph) and
from the nearest node that only exists in the future-unlabelled set (frames after the cut that are not test images of the window), and how does the per-judgment gain of a transductive learner over BASE
depend on that distance?
  python3 td_mechanism.py --dirs dev1,dev2 --learner L:trans_win --seeds 10500-10519 --out results/td/mechanism_dev.csv
Per judgment: dacc = correct(learner) - correct(BASE); dll = logloss(BASE) - logloss(learner) (positive = better)."""
import argparse, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, rt_core as rt, td_core as td

RES = Path(__file__).resolve().parent / "results" / "td"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", required=True); ap.add_argument("--learner", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cuts", default=",".join(map(str, td.CUTS))); a = ap.parse_args(); torch.set_num_threads(1)
    lo, hi = a.seeds.split("-"); cache = core.feature_cache(); sim = rt.Sim(cache); rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for seed in range(int(lo), int(hi) + 1):
            for cut in a.cuts.split(","):
                recs = {}
                for d in a.dirs.split(","):
                    f = RES / d / f"seed{seed}_cut{cut}.json"
                    if f.exists():
                        for r in json.loads(f.read_text()):
                            if r["unit"] != "meta": recs[(r["learner"], r["window"])] = r
                if ("BASE", "FA") not in recs or (a.learner, "FA") not in recs: continue
                ctx = td.make_forward_ctx(seed, int(cut), scratch, cache); tests = td.forward_tests(ctx); sets = td.exclusion_sets(ctx)
                allp = sorted(cache); ind_nodes = [p for p in allp if td.img_time(p) is None or td.img_time(p) <= ctx.forward["cut_time"]]
                held = td.heldout_paths(ctx)
                for w, t in tests.items():
                    end = td.group_time(ctx, (ctx.forward["near"] if w == "FA" else ctx.forward["far"])[-1]); fut = [p for p in allp if td.img_time(p) is not None and ctx.forward["cut_time"] < td.img_time(p) <= end and p not in held]
                    nodes_ind = [p for p in ind_nodes if p not in held]
                    pa = 1 - sim.max_sim(list(t.img1), nodes_ind); pb = 1 - sim.max_sim(list(t.img2), nodes_ind); fa = 1 - sim.max_sim(list(t.img1), fut); fb = 1 - sim.max_sim(list(t.img2), fut)
                    db, dc = np.array(recs[("BASE", w)]["d"]), np.array(recs[(a.learner, w)]["d"]); s = t.sign
                    for j in range(len(s)):
                        rows.append({"seed": seed, "cut": cut, "window": w, "nov_past": max(pa[j], pb[j]), "nov_fut": max(fa[j], fb[j]), "dacc": float((dc[j] * s[j] > 0)) - float((db[j] * s[j] > 0)),
                                     "dll": float(np.logaddexp(0, -s[j] * db[j]) - np.logaddexp(0, -s[j] * dc[j])), "far_anchor": None})
    df = pd.DataFrame(rows); a.out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False); print(len(df), "judgments ->", a.out)


if __name__ == "__main__": main()

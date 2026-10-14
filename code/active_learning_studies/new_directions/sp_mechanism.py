"""Mechanism check for the HTR route: is the gain carried by HTR test judgments whose images have a near-duplicate labelled frame?

  python3 sp_mechanism.py --dirs results/sp/dev1 results/sp/dev2 --cand rt_c2_lam10_rho0.001 [--splits A,B]

For every stored cell the labelled set is regenerated (``core.random_labelled``), the distance (1 - cos, centred/normalised SimCLR features) of each HTR test image to the nearest labelled-judgment image is computed, and the per-judgment correctness of the
head (``lap3_k10_st4``) and of the candidate is compared by terciles of the larger of the two image distances.  Descriptive; pooled over cells; the unit of the interval is the seed (bootstrap over seeds).
"""
from __future__ import annotations

import argparse
import glob
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys

import nd_core as core
import rt_core as rt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pair_disjoint_not_image_disjoint' / 'graph_exploration'))
import rc_tables  # noqa: E402  (markdown helper)

HEAD = "lap3_k10_st4"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out")
    a = ap.parse_args(); cache = core.feature_cache(); sim = rt.Sim(cache); rows = []
    files = sorted(f for d in a.dirs for f in glob.glob(str(Path(d) / "seed*.json")))
    with tempfile.TemporaryDirectory() as scratch:
        for f in files:
            blob = json.load(open(f)); recs = blob["records"]; seed, split = recs[0]["seed"], recs[0]["split"]
            if split not in a.splits.split(","): continue
            ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); m = blob["meta"]; k = np.array(m["k"]); s = np.array(m["sign"]); htr = k == 4
            by = {(r["budget"], r["draw"], r["learner"]): np.array(r["d"]) for r in recs}
            for (b, dr, name), dh in by.items():
                if name != HEAD or (b, dr, a.cand) not in by: continue
                lab = core.random_labelled(ctx, b, dr); train = rt.labelled_images(ctx, lab); da = 1 - sim.max_sim(list(test.img1), train); db = 1 - sim.max_sim(list(test.img2), train); far_img = np.maximum(da, db)
                dc = by[(b, dr, a.cand)]
                for j in np.where(htr)[0]: rows.append({"seed": seed, "split": split, "dist": far_img[j], "head_ok": float(dh[j] * s[j] > 0), "cand_ok": float(dc[j] * s[j] > 0)})
    df = pd.DataFrame(rows); df["tercile"] = pd.qcut(df.dist, 3, labels=["nearest third", "middle third", "farthest third"])
    out = []
    for t, g in df.groupby("tercile", observed=True):
        per = g.groupby("seed").apply(lambda x: (x.cand_ok - x.head_ok).mean(), include_groups=False).to_numpy(); rng = np.random.default_rng(0); boots = rng.choice(per, size=(2000, len(per))).mean(1)
        out.append({"tercile of HTR test-image distance to nearest labelled image": t, "distance range (1-cos)": f"{g.dist.min():.4f}-{g.dist.max():.4f}", "n judgments": len(g), "head accuracy": g.head_ok.mean(), "candidate accuracy": g.cand_ok.mean(), "gain": per.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
    text = rc_tables.md(pd.DataFrame(out))
    print(text) if not a.out else Path(a.out).write_text(text + "\n")


if __name__ == "__main__": main()

"""Audit 1b: for every test judgment of a (seed, split): distance (1 - cos) of each of its two images to the nearest typed ideal-image anchor (references / utility / outer-test ideal images, i.e. the
anchors of aw8_all) and to the nearest labelled image is NOT needed here (see a1).  Output: one CSV per (seed, split), keyed like a1 by (seed, split, j).
  python3 rt_a1b.py --seeds 8800-8819 --splits A,B --out results/rt/a1b"""
import argparse, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, rt_core as rt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8819"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args(); torch.set_num_threads(1)
    lo, hi = args.seeds.split("-"); cache = core.feature_cache(); sim = rt.Sim(cache); args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in range(int(lo), int(hi) + 1):
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.csv"
                if target.exists(): continue
                ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); out = {"seed": seed, "split": split, "j": np.arange(len(test.k)), "k": test.k}
                for tag, table in (("ref", ctx.exp.references), ("util", ctx.exp.utility_images), ("outer", ctx.exp.test_images)):
                    anchors = sorted({str(p) for ps in table.values() for p in ps}); da, db = 1 - sim.max_sim(list(test.img1), anchors), 1 - sim.max_sim(list(test.img2), anchors)
                    out[f"anc_{tag}_min"] = np.minimum(da, db); out[f"anc_{tag}_max"] = np.maximum(da, db)
                allanc = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps}); da, db = 1 - sim.max_sim(list(test.img1), allanc), 1 - sim.max_sim(list(test.img2), allanc)
                out["anc_all_min"], out["anc_all_max"] = np.minimum(da, db), np.maximum(da, db); pd.DataFrame(out).to_csv(target, index=False)


if __name__ == "__main__": main()

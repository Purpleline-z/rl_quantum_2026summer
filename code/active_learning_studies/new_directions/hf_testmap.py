"""HTR failure analysis: map the held-out decisive judgments of every (seed, split) (the rows of ``nd_core.TestSet`` in order) to the global judgment id of ``hf_table.py``.

  python3 hf_testmap.py --seeds 10400-10419 --out results/hf/testmap_dev.csv     columns: seed, split, j, jid, plus split role flags (test group, in validation, etc.)
"""
import argparse, tempfile
from pathlib import Path
import pandas as pd
import nd_core as core


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    judg = pd.read_csv(core.HERE / "results" / "hf" / "judgments.csv"); root = str(core.single_study.DATA.resolve()) + "/original data/"
    key = {(r.img1, r.img2, int(r.type_idx)): int(r.jid) for r in judg.itertuples()}; cache = core.feature_cache(); rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds_of(args.seeds):
            for split in args.splits.split(","):
                ctx = core.make_ctx(seed, split, scratch, cache); t = core.TestSet(ctx)
                for j, (a, b, k) in enumerate(zip(t.img1, t.img2, t.k)): rows.append({"seed": seed, "split": split, "j": j, "jid": key[(str(a).replace(root, ""), str(b).replace(root, ""), int(k))]})
    args.out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(args.out, index=False); print(len(rows), "rows ->", args.out)


if __name__ == "__main__": main()

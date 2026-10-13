"""Learning-curve tables (metric vs judgment budget with 95% bootstrap bands over seeds) from the stored confirmation cells; random selection only; mean over the 3 label draws per (seed, budget).

Block N (default-head ladder; seeds 3000-3034 and 5000-5034, 70 seeds): baseline (default head), aw8, aw8_all.
Block S (current learner ladder; seeds 8300-8334 and 8350-8384, 70 seeds): aw8_all, aw8_all_st2lr0.5 (tuned graph-free), lap3_k10, lap3_k10_st4, lap3_k10_st4_shuf (shuffled-graph control).
The two blocks have different seeds and therefore different held-out sets; aw8_all is present in both and links them.  Paired differences (to aw8_all) are always taken inside one block.
Outputs: tables/curves.csv (levels), tables/curves_diff.csv (paired difference to aw8_all, by budget), tables/curves_type.csv (per type level and difference), numbers_curves.json."""
import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import Numbers

BLOCKS = {"N": (["confirm1", "confirm4_htr"], ["baseline", "aw8", "aw8_all"], "seeds 3000-3034 + 5000-5034"),
          "S": (["sh/c1_learn", "sh/c2_learn"], ["aw8_all", "aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4", "lap3_k10_st4_shuf"], "seeds 8300-8334 + 8350-8384")}
METRICS = ["auc", "acc", "ll"]; TYPES = ["t1x1_acc", "tc6x2_acc", "t13_acc", "htr_acc"]


def block_frame(rels):
    f = pd.concat([L.load_dir(r) for r in rels], ignore_index=True); return f[f.selector == "random"]


def main():
    N = Numbers("curves"); lev, dif, typ = [], [], []
    for blk, (rels, learners, desc) in BLOCKS.items():
        f = block_frame(rels)
        for split in ("A", "B"):
            for l in learners:
                for m in METRICS + TYPES:
                    t = L.level_by_budget(f, split, m, ("random", l))
                    for b in t.columns:
                        x = t[b].dropna(); lo, hi = L.boot_ci(x.to_numpy()); row = dict(block=blk, learner=l, split=split, metric=m, budget=int(b), mean=float(x.mean()), lo=lo, hi=hi, n_seeds=int(len(x)))
                        (typ if m in TYPES else lev).append(row)
                        if m in ("auc", "acc", "ll", "htr_acc"): N.put(f"cv.{blk}.{l}.{split}.{m}.b{int(b)}", row["mean"])
                    if l != "aw8_all":
                        for b, d in L.per_seed(f, split, m, ("random", l), ("random", "aw8_all"), by_budget=True).groupby(level="budget"):
                            x = d.droplevel("budget"); lo, hi = L.boot_ci(x.to_numpy()); row = dict(block=blk, learner=l, ref="aw8_all", split=split, metric=m, budget=int(b), gain=float(x.mean()), lo=lo, hi=hi, n_seeds=int(len(x)), p=L.wilcoxon_p(x))
                            (typ if m in TYPES else dif).append(row)
                            if m in ("auc", "acc", "htr_acc"): N.put(f"cvd.{blk}.{l}.{split}.{m}.b{int(b)}", row["gain"])
    pd.DataFrame(lev).to_csv(L.TABLES / "curves.csv", index=False); pd.DataFrame(dif).to_csv(L.TABLES / "curves_diff.csv", index=False); pd.DataFrame(typ).to_csv(L.TABLES / "curves_type.csv", index=False)
    # label-efficiency numbers: does the regularised learner at budget 10 reach aw8_all at budget 60?
    lv = pd.DataFrame(lev)
    for split in ("A", "B"):
        for m in ("auc", "acc"):
            g = lambda l, b: float(lv[(lv.block == "S") & (lv.learner == l) & (lv.split == split) & (lv.metric == m) & (lv.budget == b)]["mean"].iloc[0])
            N.put(f"cv.S.st4_b10_minus_aw8all_b60.{split}.{m}", g("lap3_k10_st4", 10) - g("aw8_all", 60)); N.put(f"cv.S.st4_b20_minus_aw8all_b60.{split}.{m}", g("lap3_k10_st4", 20) - g("aw8_all", 60))
    N.save(); print("curves:", len(lev), len(dif), len(typ))


if __name__ == "__main__":
    main()

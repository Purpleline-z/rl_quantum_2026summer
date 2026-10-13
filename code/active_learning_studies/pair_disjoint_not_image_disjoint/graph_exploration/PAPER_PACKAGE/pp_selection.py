"""Selection rules vs random selection as a function of learner strength (single-shot selection from the model trained on the initial groups; random = mean of 3 draws; paired on identical initial model and pool).
Rows: (learner, selector) cell minus (learner, random) cell, per split, AUC / accuracy / log-loss (+ HTR metrics), paired bootstrap interval over seeds, uncorrected Wilcoxon p; the Holm values of the
pre-registered families are in confirmations.csv.  Blocks: ND1 (3000-3034, learners baseline and aw8, rule vopt_u), ND3 (4000-4034, cold start), SH1/SH2 (8300-8334, 8350-8384, learner lap3_k10, four candidates and controls),
SH1S4 (8300-8334, learner lap3_k10_st4).   Outputs tables/selection.csv, tables/selection.md, numbers_selection.json (keys sel.<block>.<learner>.<selector>.<split>.<metric>.<field>)."""
import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import Numbers

JOBS = [("ND1", "confirm1", "baseline", ["vopt_u"]), ("ND1", "confirm1", "aw8", ["vopt_u"]), ("ND1", "confirm1", "aw8_all", ["vopt_u"]),
        ("ND3", "confirm3_cold", "baseline", ["vopt_u"]), ("ND3", "confirm3_cold", "aw8", ["vopt_u"]),
        ("SH1", "sh/c1_sel", "lap3_k10", ["vopt_u", "gvopt_node3", "gvopt_node3_shuf", "typed_decisive_bald", "typed_decisive_bald_shuffled", "vopt_htr4"]),
        ("SH2", "sh/c2_sel", "lap3_k10", ["vopt_u", "gvopt_node3", "gvopt_node3_shuf", "typed_decisive_bald", "typed_decisive_bald_shuffled", "vopt_htr4"]),
        ("SH1S4", "sh/c1_sel4", "lap3_k10_st4", ["vopt_u", "gvopt_node3", "gvopt_node3_shuf", "typed_decisive_bald", "typed_decisive_bald_shuffled", "vopt_htr4"])]
METRICS = ["auc", "acc", "ll", "htr_acc", "htr_ll"]


def main():
    N = Numbers("selection"); rows = []
    for blk, rel, learner, sels in JOBS:
        f = L.load_dir(rel)
        for sel in sels:
            for split in ("A", "B"):
                for m in METRICS:
                    if sel not in set(f.selector): continue
                    per = L.per_seed(f, split, m, (sel, learner), ("random", learner)); s = L.summarize(per)
                    rows.append(dict(block=blk, learner=learner, selector=sel, split=split, metric=m, **{k: s[k] for k in ("gain", "lo", "hi", "share_better", "p", "n")}))
                    for fld in ("gain", "lo", "hi", "p"): N.put(f"sel.{blk}.{learner}.{sel}.{split}.{m}.{fld}", s[fld])
    t = pd.DataFrame(rows); t.to_csv(L.TABLES / "selection.csv", index=False)
    show = t[t.metric.isin(["auc", "acc"])].copy(); show["gain [95% CI]"] = [L.fmt_gain(g, lo, hi) for g, lo, hi in zip(show.gain, show.lo, show.hi)]
    L.write(L.TABLES / "selection.md", "| block | learner | selector | split | metric | gain over random [95% CI] | p (uncorrected) |\n|---|---|---|---|---|---|---|\n" + "\n".join(f"| {r.block} | {r.learner} | {r.selector} | {r.split} | {r.metric} | {r['gain [95% CI]']} | {L.fmt_p(r.p)} |" for _, r in show.iterrows()) + "\n")
    # composition of the HTR-directed rule: share of picks on HTR (descriptive)
    for blk, rel in (("SH1", "sh/c1_sel"), ("SH2", "sh/c2_sel")):
        f = L.load_dir(rel); g = f[f.learner == "lap3_k10"].groupby("selector")[["n_type0", "n_type2", "n_type3", "n_type4"]].mean(); share = g.n_type4 / g.sum(axis=1)
        for s_ in ("random", "vopt_htr4", "vopt_u"): N.put(f"sel.{blk}.htr_share.{s_}", float(share[s_]))
    N.save(); print(len(t), "selection rows")


if __name__ == "__main__":
    main()

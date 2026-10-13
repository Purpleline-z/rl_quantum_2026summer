"""Learner ablation ladder: gain of every step with 95% bootstrap interval (paired over seeds, same label sets), from the stored cells of the blocks in which the step was pre-registered / replicated.
Steps are measured on different seed blocks (each step is paired inside its own block); the 'sum of steps' row adds the step means and combines independent bootstrap intervals (indicative only).
  L1 default head -> aw8 (anchor weight 0.25 -> 8)             ND1 (3000-3034, pre-registered), ND4 (5000-5034, replication)
  L2 aw8 -> aw8_all (all ideal images as anchors)               descriptive in ND4 (5000-5034) and ND1 (3000-3034)
  L3 aw8_all -> lap3_k10 (Laplacian regulariser, lambda 3)      GL1 (7100-7134), GL2 (7200-7234); descriptive replications in GM1/GM2, SH1/SH2 cells
  L4 lap3_k10 -> lap3_k10_st4 (4x training steps)               SH1 (8300-8334), SH2 (8350-8384), SHC cold start (8385-8399)
Controls: aw8_all -> aw8_all_st2lr0.5 (graph-free schedule tuning), lap3_k10_st4 vs aw8_all_st2lr0.5 (pre-registered primary), lap3_k10_st4 vs its shuffled-graph control, lap3_k10 vs lap3_k10_shuf (GL).
Outputs tables/ladder.csv, tables/ladder.md, numbers_ladder.json (keys ld.<step>.<block>.<split>.<metric>.<field>)."""
import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import Numbers

R = "random"
STEPS = [  # (id, label, dirs, a-cell, b-cell, block-id, seeds, status)
    ("L1", "default head -> aw8", ["confirm1"], "aw8", "baseline", "ND1", "3000-3034", "pre-registered"), ("L1", "default head -> aw8", ["confirm4_htr"], "aw8", "baseline", "ND4", "5000-5034", "descriptive replication (ND4 spec tested HTR metrics only)"),
    ("L2", "aw8 -> aw8_all", ["confirm4_htr"], "aw8_all", "aw8", "ND4", "5000-5034", "descriptive (not a frozen test)"), ("L2", "aw8 -> aw8_all", ["confirm1"], "aw8_all", "aw8", "ND1", "3000-3034", "descriptive (difference of two frozen contrasts)"),
    ("L3", "aw8_all -> lap3_k10", ["gl/confirm1"], "lap3_k10", "aw8_all", "GL1", "7100-7134", "pre-registered"), ("L3", "aw8_all -> lap3_k10", ["gl/confirm2"], "lap3_k10", "aw8_all", "GL2", "7200-7234", "replication (pre-registered)"),
    ("L3", "aw8_all -> lap3_k10", ["gl/cold"], "lap3_k10", "aw8_all", "GLC", "7300-7334", "cold start (pre-registered)"),
    ("L3", "aw8_all -> lap3_k10", ["gm/confirm1"], "lap3_k10", "aw8_all", "GM1", "8100-8134", "descriptive (same cells as the GM1 reference)"), ("L3", "aw8_all -> lap3_k10", ["gm/confirm2"], "lap3_k10", "aw8_all", "GM2", "8150-8184", "descriptive"),
    ("L3", "aw8_all -> lap3_k10", ["sh/c1_learn"], "lap3_k10", "aw8_all", "SH1", "8300-8334", "descriptive"), ("L3", "aw8_all -> lap3_k10", ["sh/c2_learn"], "lap3_k10", "aw8_all", "SH2", "8350-8384", "descriptive"),
    ("L4", "lap3_k10 -> lap3_k10_st4", ["sh/c1_learn"], "lap3_k10_st4", "lap3_k10", "SH1", "8300-8334", "pre-registered (secondary family)"), ("L4", "lap3_k10 -> lap3_k10_st4", ["sh/c2_learn"], "lap3_k10_st4", "lap3_k10", "SH2", "8350-8384", "replication (pre-registered)"),
    ("L4", "lap3_k10 -> lap3_k10_st4", ["sh/cold_learn"], "lap3_k10_st4", "lap3_k10", "SHC", "8385-8399", "cold start (pre-registered)"),
    ("T", "aw8_all -> lap3_k10_st4 (recipe, direct)", ["sh/c1_learn"], "lap3_k10_st4", "aw8_all", "SH1", "8300-8334", "descriptive"), ("T", "aw8_all -> lap3_k10_st4 (recipe, direct)", ["sh/c2_learn"], "lap3_k10_st4", "aw8_all", "SH2", "8350-8384", "descriptive"),
    ("C1", "control: aw8_all -> aw8_all_st2lr0.5 (graph-free tuning)", ["sh/c1_learn"], "aw8_all_st2lr0.5", "aw8_all", "SH1", "8300-8334", "descriptive"), ("C1", "control: aw8_all -> aw8_all_st2lr0.5 (graph-free tuning)", ["sh/c2_learn"], "aw8_all_st2lr0.5", "aw8_all", "SH2", "8350-8384", "descriptive"),
    ("C2", "primary: lap3_k10_st4 vs tuned graph-free aw8_all_st2lr0.5", ["sh/c1_learn"], "lap3_k10_st4", "aw8_all_st2lr0.5", "SH1", "8300-8334", "pre-registered primary"), ("C2", "primary: lap3_k10_st4 vs tuned graph-free aw8_all_st2lr0.5", ["sh/c2_learn"], "lap3_k10_st4", "aw8_all_st2lr0.5", "SH2", "8350-8384", "replication (pre-registered)"),
    ("C2", "primary: lap3_k10_st4 vs tuned graph-free aw8_all_st2lr0.5", ["sh/cold_learn"], "lap3_k10_st4", "aw8_all_st2lr0.5", "SHC", "8385-8399", "cold start (pre-registered)"),
    ("C3", "control: lap3_k10_st4 vs its shuffled-graph control", ["sh/c1_learn"], "lap3_k10_st4", "lap3_k10_st4_shuf", "SH1", "8300-8334", "pre-registered control (uncorrected)"), ("C3", "control: lap3_k10_st4 vs its shuffled-graph control", ["sh/c2_learn"], "lap3_k10_st4", "lap3_k10_st4_shuf", "SH2", "8350-8384", "control"),
    ("C4", "control: lap3_k10 vs its shuffled-graph control", ["gl/confirm1"], "lap3_k10", "lap3_k10_shuf", "GL1", "7100-7134", "pre-registered control (uncorrected)"), ("C4", "control: lap3_k10 vs its shuffled-graph control", ["gl/confirm2"], "lap3_k10", "lap3_k10_shuf", "GL2", "7200-7234", "control"),
]


def main():
    N = Numbers("ladder"); rows = []; per_cache = {}
    for sid, label, dirs, a, b, blk, seeds, status in STEPS:
        f = pd.concat([L.load_dir(d) for d in dirs], ignore_index=True); f = f[f.selector == R]
        for split in ("A", "B"):
            for m in ("auc", "acc", "ll"):
                per = L.per_seed(f, split, m, (R, a), (R, b)); s = L.summarize(per); per_cache[(sid, blk, split, m)] = per
                rows.append(dict(step=sid, label=label, block=blk, seeds=seeds, status=status, split=split, metric=m, a=a, b=b, **{k: s[k] for k in ("gain", "lo", "hi", "share_better", "p", "n")}))
                for fld in ("gain", "lo", "hi", "p", "n"): N.put(f"ld.{sid}.{blk}.{split}.{m}.{fld}", s[fld])
    # pooled estimates over the pre-registered blocks of a step (per-seed gains of different blocks concatenated; bootstrap over seeds); descriptive
    POOLS = [("L3", "aw8_all -> lap3_k10 (pooled GL1+GL2)", [("GL1", "gl/confirm1"), ("GL2", "gl/confirm2")], "lap3_k10", "aw8_all"), ("L4", "lap3_k10 -> lap3_k10_st4 (pooled SH1+SH2)", [("SH1", "sh/c1_learn"), ("SH2", "sh/c2_learn")], "lap3_k10_st4", "lap3_k10"),
             ("C2", "lap3_k10_st4 vs tuned graph-free (pooled SH1+SH2)", [("SH1", "sh/c1_learn"), ("SH2", "sh/c2_learn")], "lap3_k10_st4", "aw8_all_st2lr0.5"),
             ("C2c", "lap3_k10_st4 vs tuned graph-free (pooled SH1+SH2+SHC cold start)", [("SH1", "sh/c1_learn"), ("SH2", "sh/c2_learn"), ("SHC", "sh/cold_learn")], "lap3_k10_st4", "aw8_all_st2lr0.5")]
    for pid, label, blocks, a, b in POOLS:
        for split in ("A", "B"):
            for m in ("auc", "acc", "ll"):
                per = pd.concat([L.per_seed(L.load_dir(rel).query("selector == @R"), split, m, (R, a), (R, b)) for _, rel in blocks]); s_ = L.summarize(per)
                rows.append(dict(step="POOL-" + pid, label=label, block="+".join(x for x, _ in blocks), seeds=f"{len(per)} seeds", status="pooled over pre-registered blocks (descriptive)", split=split, metric=m, a=a, b=b, **{k: s_[k] for k in ("gain", "lo", "hi", "share_better", "p", "n")}))
                for fld in ("gain", "lo", "hi", "n"): N.put(f"ld.POOL.{pid}.{split}.{m}.{fld}", s_[fld])
    t = pd.DataFrame(rows)
    # indicative sum of the four steps (primary block of each step): L1 ND1, L2 ND4, L3 GL1, L4 SH1; bootstrap each step's per-seed gains independently and add
    rng = np.random.default_rng(1); prim = [("L1", "ND1"), ("L2", "ND4"), ("L3", "GL1"), ("L4", "SH1")]; srows = []
    for split in ("A", "B"):
        for m in ("auc", "acc", "ll"):
            draws = sum(rng.choice(per_cache[(s, bl, split, m)].to_numpy(), size=(L.N_BOOT, len(per_cache[(s, bl, split, m)]))).mean(1) for s, bl in prim); mean = sum(float(per_cache[(s, bl, split, m)].mean()) for s, bl in prim)
            lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))); srows.append(dict(step="SUM", label="sum of L1+L2+L3+L4 (primary blocks; indicative: different seed blocks)", block="ND1+ND4+GL1+SH1", seeds="4 blocks", status="indicative", split=split, metric=m, a="lap3_k10_st4", b="default head", gain=mean, lo=lo, hi=hi, share_better=np.nan, p=np.nan, n=0))
            for fld, v in (("gain", mean), ("lo", lo), ("hi", hi)): N.put(f"ld.SUM.{split}.{m}.{fld}", v)
    t = pd.concat([t, pd.DataFrame(srows)], ignore_index=True); t.to_csv(L.TABLES / "ladder.csv", index=False)
    show = t[t.metric.isin(["auc", "acc"])].copy(); show["gain [95% CI]"] = [L.fmt_gain(g, lo, hi) for g, lo, hi in zip(show.gain, show.lo, show.hi)]; show["p"] = [("" if np.isnan(p) else L.fmt_p(p)) for p in show.p]
    L.write(L.TABLES / "ladder.md", "| step | block (seeds) | status | split | metric | gain [95% CI] | p (uncorrected) |\n|---|---|---|---|---|---|---|\n" + "\n".join(f"| {r.step}: {r.label} | {r.block} ({r.seeds}) | {r.status} | {r.split} | {r.metric} | {r['gain [95% CI]']} | {r.p} |" for _, r in show.iterrows()) + "\n")
    N.save()


if __name__ == "__main__":
    main()

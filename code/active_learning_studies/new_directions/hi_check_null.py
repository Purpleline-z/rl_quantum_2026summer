"""Development-data check that motivated AMENDMENT_A1 (run on the development block only; no confirmation data):
 (i) the null SD of the exact swap-randomisation statistic from the linear formula equals the SD of directly recomputed random swaps (the test is exact);
 (ii) but it is ~1.6-1.8x the empirical SE of the estimate from random half-splits of the 168 groups (SE_full ~ sd(est_halfA - est_halfB) / 2), which agrees with the group bootstrap SE within ~10 %."""
import sys
import numpy as np

import hi_analyze as A
import hi_infer as H

blk = A.load(["results/hi/dev"]); ug = A.unit_of(blk, "unit_g"); rng = np.random.default_rng(3)
for comp, cand, ref, split in [("C3", "lap3_k10", "aw8_all", "A"), ("C5", "lap3_k10_st4", "aw8_all_st2lr0.5", "A"), ("C1", "aw8", "baseline", "B")]:
    cm = (blk.cells.split == split).to_numpy(); full = H.Design(blk, cand, ref, "auc", cell_mask=cm); pc = full.perm_contributions(); null_sd = float(np.sqrt((pc ** 2).sum()))
    c, r = blk.D[cand].copy(), blk.D[ref].copy(); direct = []
    for i in range(100):
        sw = (rng.random(blk.n_groups) < 0.5)[blk.grp]; blk.D["x"], blk.D["y"] = np.where(sw, r, c), np.where(sw, c, r); direct.append(H.Design(blk, "x", "y", "auc", cell_mask=cm).estimate())
    ds = []
    for i in range(300):
        h = rng.permutation(blk.n_groups) < blk.n_groups // 2; m = h[blk.grp]; ds.append(H.Design(blk, cand, ref, "auc", mask=m, cell_mask=cm).estimate() - H.Design(blk, cand, ref, "auc", mask=~m, cell_mask=cm).estimate())
    print(f"{comp} split {split}: swap-null SD formula {null_sd:.5f}  direct swaps {np.std(direct):.5f}  half-split SE {np.nanstd(ds) / 2:.5f}  group-bootstrap SE {np.nanstd(H.group_bootstrap(full, ug, B=1000, seed=0)):.5f}", flush=True)

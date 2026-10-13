# Pre-registration TG1 (frozen before any confirmation seed) — temporal edges for the Laplacian learner

Written 2026-10-13 ~18:05 UTC after the development screens dev1 (seeds 9200–9209), dev2 (9210–9219), devb (time-blocked split, 9200–9219), see `PREREGISTRATION_TG0_development_plan.md` and `TG1_dev_tables.md` (script-generated). **Confirmation seeds 9250–9284 (35), splits A and B, one draw per (seed, split, budget) — no confirmation cell exists at the time of this commit.** Each cell is run once; nothing is re-tuned afterwards. Run script: `scripts_tg/chain_confirm.sh`; test list: `python3 tg_tables.py confirm TG1_spec.json DIR [BLOCKED_DIR]`.

## What development showed (20 development seeds, optimistic; 11 + 7 variants screened, so winner's-curse applies)
1. **No temporal variant improves the incumbent `lap3_k10_st4`** in a way that survives its controls. Most are null or worse on AUC/accuracy, and nearly all lose on the anchor-far endpoint and on log-loss: temperature-bin edges −0.16 AUC (their partners are far apart in feature space), temporal-only graph −0.02, non-kNN chain edges −0.007/−0.011 AUC and −0.055 log-loss, intersection (dropping the ≈4 % cross-run / far-in-time kNN edges) −0.007/−0.006 AUC.
2. **84–94 % of the chain edges are already kNN edges** (graph-statistics table), so the temporal term mostly re-weights existing edges.
3. The one variant with a consistent positive sign over the 20 development seeds is the decayed chain ±5 (`tg_dec5`): AUC +0.0020 (A) / +0.0016 (B), accuracy +0.0002 / +0.0039 (both far below any individual significance after correction). Its gentle sibling (mu 0.3) and the second-difference penalty `tg_curv` (distinct mechanism; the only one with an AUC gain in the red-team blocked split in development, +0.004) are frozen with it.
4. Red-team blocked split (development, 20 seeds): AUC +0.001…+0.004 for chain/curvature variants, accuracy ≈ 0 or negative, log-loss worse; the "inb" stratum (both test images inside a held-out block) shows nothing. So the hoped-for larger benefit on time-blocked data is at most a very small AUC effect.
5. Far-in-time kNN edge re-weighting (`tg_xr*`) and the other screens: null.

## Frozen candidates (3) and controls
All candidates = `lap3_k10_st4` + a temporal term, mu = weight of the temporal term (kNN term unchanged, λ = 3), 256 sampled edges per step from an independent random stream; definitions in `nd_temporal_graph.py` (`SPEC_OF`):
- **C1 `tg_dec5`**: chain edges between frames of the same run with index gap ≤ 5, edge weights exp(−gap/2)·exp(−|ΔT|/30 °C), mu = 3. Control `tg_dec5_shuf` (endpoints permuted within each run, degree preserved).
- **C2 `tg_dec5_mu0.3`**: the same with mu = 0.3. Control `tg_dec5_mu0.3_shuf`.
- **C3 `tg_curv`**: second-difference penalty (s(i−1) − 2 s(i) + s(i+1))² along each run (consecutive frames), mu = 3. Control `tg_curv_shuf`.
Reference (baseline): `lap3_k10_st4` on the identical random label sets. Learners are paired on the same label sets (seed, budget).

## Tests and success rule
- **Main family (Holm over 3 candidates × 2 splits × {AUC, accuracy} = 12 tests):** per-seed mean over budgets of (candidate − baseline), two-sided Wilcoxon over the 35 seeds; Holm step-down.
- **Success of a candidate (the study's rule):** AUC and accuracy gains positive in both splits (all 4 differences > 0), ≥ 3 of the 4 Holm-adjusted p < 0.05, and better than its shuffled-time control (candidate − control > 0 with uncorrected p < 0.05 in ≥ 1 of the 4 split × metric cells; the control comparison is uncorrected).
- **Anchor-far secondary endpoint (reported for every test above, uncorrected):** `far_auc`, `far_acc` = metrics on test judgments whose two images are BOTH > 0.03 cosine distance from every typed ideal-image anchor (`rt_a1b.py` definition; ≈ 30 % of the test judgments); `near_*` = complement. **Honesty clause:** if a candidate passes the main rule but its far gains are not positive in both splits, the result is reported as an anchor-near gain (not a general gain); a pass is qualified as "anchor-far supported" only if far_auc and far_acc gains are positive in both splits.
- **Time-blocked split (secondary, uncorrected, descriptive):** same candidates and baseline on the red-team blocked contexts (`rt_blocked.make_blocked_ctx`, one context per seed, seeds 9250–9284), AUC, accuracy, `far_*`, `inb_*` (both images inside a held-out block; the 25-frame buffer holds only there), log-loss. No success rule; the unit-tested property is that no graph node lies within 25 frames of an in-block test image. The strict variant (temporal edges dropped within 25 frame numbers of any held-out image) is not a candidate: in the standard splits it removes ≈ 98 % of the temporal edges (measured 34 of 1766 for chain ±2), i.e. it is the baseline.
- Other secondaries (not in any rule): log-loss, HTR accuracy, gain by budget.

## Expectation stated in advance
All three candidates are expected to be **not confirmed**: the development effects are ≈ +0.002 AUC and ≈ 0 accuracy, well inside what 35 seeds can resolve (the confirmed effects of the earlier learner studies were +0.006…+0.01 AUC), and temporal edges largely duplicate the kNN graph. A pass of C1/C2 would have to be read together with the anchor-far endpoint.

## Limits
The 168 labelled pair groups are the same in every seed (re-splitting, not new data), so p-values are optimistic and development selection already used 20 seeds; one draw per cell; the shuffled-time control is harsh (random within-run pairs are far apart in feature space), so beating it is weak evidence for temporal order — the local-shuffle and random-kNN-edge controls of development are not part of the success rule.

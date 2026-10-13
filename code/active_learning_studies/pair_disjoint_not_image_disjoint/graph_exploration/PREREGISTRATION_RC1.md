# Pre-registration RC1 (recipe search) — frozen 2026-10-13 ≈ 18:20, before any confirmation seed (9350–9384) was run

Development: seeds 9300–9319 (`PREREGISTRATION_RC0_development_plan.md` + amendment RC0b); code `new_directions/nd_recipe.py`, `rc_run.py`; tables `rc_tables.py`, analysis `rc_confirm.py`; spec `RC1_spec.json`.

## What development showed (observations, not evidence)
Twelve single-knob variants (tier 1, 10 seeds × 2 splits × 4 budgets × 2 random label sets) and one combination (20 seeds) were screened against the unchanged incumbent `lap3_k10_st4`. **No variant improved on it**: all AUC / accuracy gains are within about ±0.005 (20-seed extensions: AUC −0.0013 … +0.0018, accuracy −0.0027 … +0.0004; no uncorrected p < 0.03 in the favourable direction); dropout 0.3 active in training is clearly harmful (AUC ≈ −0.12), hinge anchors cost accuracy (−0.017/−0.004) and log-loss (−0.05/−0.04), the graph-free control `aw8_all_st2lr0.5` is worse by −0.022/−0.017 AUC (reproducing the earlier confirmations on new seeds). The between-seed SD of the paired gain is ≈ 0.005 AUC / 0.008 accuracy, so 35 confirmation seeds can detect ≈ +0.003 AUC / +0.005 accuracy at the Holm level (`dev_mde` table in `RESULTS_recipe.md`). The promotion rule of RC0 (≥ 3 of 4 gains positive and mean > +0.002) promoted nothing; per RC0b the ranking-score rule produced the candidates below, **and they are confirmed anyway as a direct winner's-curse test**. The expected outcome is "no gain"; a surprise would be a pre-registered success of a variant whose development gain was ≈ 0.

## Frozen candidates (4) and reference
Reference = the unchanged incumbent `lap3_k10_st4`.
1. `rc_cos` — cosine learning-rate decay to 0 (peak learning rate unchanged); 20-seed development score +0.0001 (rank 1).
2. `rc_ptlr` — final-layer update ×2 for the RT13 and HTR output rows (per-type learning rate); score −0.0005 (rank 2; nominal log-loss gain +0.002/+0.005).
3. `rc_aw16` — reference-anchor weight 16 instead of 8; score −0.0008 (rank 3; log-loss +0.009/+0.011, anchor-far gains negative).
4. `rcx_rc_cos+rc_wd1` — cosine decay + AdamW weight decay 1.0 (the combination of the two top-ranked tier-1 variants with distinct switches); score −0.0003.
Context (not part of any family): the graph-free control `aw8_all_st2lr0.5` is run on the same seeds to check that the incumbent still beats it.

## Confirmation design
Seeds 9350–9384 (35), splits A and B, budgets 10/20/40/60, 3 random label sets per budget (identical for all learners), run once. Per seed the gain is the mean over budgets and draws of the paired candidate − reference difference (log-loss: reference − candidate); two-sided Wilcoxon signed-rank over the 35 seeds; paired bootstrap interval; Holm within each family.
- **Main family** (Holm over 16 tests): 4 candidates × splits {A, B} × {AUC, accuracy} on all held-out decisive judgments. **Success of a candidate** = all four gains positive AND ≥ 3 of the 4 Holm p < 0.05.
- **HTR family** (separate; Holm over 16 tests): 4 candidates × splits × {HTR accuracy, HTR log-loss}; same rule.
- **Anchor-far secondary family** (Holm over 16 tests): judgments whose two images are both > 0.03 (1 − cos, centred/normalised SimCLR features) from every typed ideal-image anchor (`anc_all_min` of `rt_a1b.py`; ≈ 15 % of the held-out judgments, ≈ 4–15 per cell, so low power: MDE ≈ +0.013 AUC / +0.019 accuracy); AUC of a cell with a single class is undefined and skipped; same rule. A candidate that succeeds on the main family but not on the far family is reported as "gain only (or mostly) in the anchor-near stratum".
- Descriptive (uncorrected, not part of any family): per-type accuracy / log-loss gains, strata of anchor distance (≤ 0.001, 0.001–0.03, > 0.03, > 0.01), per-budget gains.
- No re-tuning after the results; failures are reported as failures.

## Runs
```
bash scripts_rc/run_rc.sh conf lap3_k10_st4,aw8_all_st2lr0.5,rc_cos,rc_ptlr,rc_aw16,rcx_rc_cos+rc_wd1 9350-9384 3
python3 rc_confirm.py ../../new_directions/results/rc/conf --cands rc_cos,rc_ptlr,rc_aw16,rcx_rc_cos+rc_wd1 --control aw8_all_st2lr0.5
```
Limits: same 168 labelled groups as all earlier work (re-split, not new data); test sets are ≈ 60–70 decisive judgments per (seed, split); the labelled data are 284 frames of three growth runs (red-team audit); the winner's-curse design above applies to the 12 + 1 screened learners.

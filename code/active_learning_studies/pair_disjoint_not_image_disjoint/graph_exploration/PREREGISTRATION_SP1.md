# Pre-registration SP1 (score propagation) — frozen 2026-10-14 ≈ 05:38, before any confirmation seed (11230–11299) was run

Development: seeds 11200–11219 (`PREREGISTRATION_SP0_development_plan.md` + amendment SP0b); code `new_directions/nd_propagate.py`, `sp_run.py`; analysis `sp_confirm.py`; spec `SP1_spec.json`; tests `code_behavior_tests/test_score_propagation.py`.

## Question
How much of the head's gain is interpolation that a non-parametric method already provides, and does a non-parametric / hybrid model beat `lap3_k10_st4` anywhere — in particular on judgments far from every typed anchor?

## Models
Reference = `lap3_k10_st4` (unchanged). Non-parametric free scores: Bradley–Terry likelihood of the labelled judgments + anchor ranking terms (weight 8, all ideal images) + bad-image term + λ·Laplacian over all graph nodes (+ extra nodes for anchors that are not graph nodes) + ρ·ridge, solved by damped Newton; test images (never nodes) score by the softmax(sim/0.05)-weighted mean of their 5 nearest nodes. Candidates (4):
1. **`rt_c2_lam10_rho0.001`** — head ⊕ free scores for HTR: HTR judgments (type index 4) use 2 × d_free (λ = 10, ρ = 0.001), all other types keep the head's score (d_head of `lap3_k10_st4`).
2. **`fs_lam10`** — pure free scores (λ = 10, ρ = 0.01), no head.
3. **`hy_lamr10_rhor0.1`** — head + free residual (λ_r = 10, ρ_r = 0.1 on the residual; likelihood and anchor terms on head + residual).
4. **`ds_mu0.03_lam10`** — the head with pair-difference distillation (μ = 0.03) of the λ = 10 free-score teacher on unlabelled nodes / edges.
Controls (not in the families; uncorrected, descriptive): `aw8_all_st2lr0.5`; shuffled graph: `rt_c2_lam10_rho0.001_sh1`, `fs_lam10_sh1`, `hy_lamr10_rhor0.1_sh1`, `ds_mu0.03_lam10_sh1` (first label set per budget only); no penalty: `rt_c2_lam0`, `fs_lam0`; baselines `kn` (nearest scored node, no Laplacian) and `ls` (label spreading of the anchors, no judgment).

## Design
Confirmation 1: seeds 11230–11264 (35), splits A and B, budgets 10/20/40/60, 3 random label sets per budget identical for all learners; run once. Secondary evaluations on the same seeds: time-blocked contexts (`rt_blocked.py`, 2 label sets per budget) and cold start (10 random initial judgments, 1 label set per budget). Confirmation 2 (independent seeds 11265–11299; standard splits only) is run only if time allows and is analysed with the same rules as a replication. Per seed the gain of a candidate over the reference is the mean over budgets and label sets of the paired difference (log-loss: reference − candidate); two-sided Wilcoxon over seeds; paired bootstrap interval.
- **Main family** (Holm over 16 tests): 4 candidates × splits {A, B} × {AUC, accuracy}, two-sided. **BETTER** = all four gains > 0 and ≥ 3 of 4 Holm p < 0.05; **WORSE** = all four < 0 and ≥ 3 of 4 Holm p < 0.05.
- **Non-inferiority family** (Holm over the same 16 cells, one-sided Wilcoxon on gain + margin): margin 0.003 AUC and 0.01 accuracy; **NON-INFERIOR** (if not BETTER) = ≥ 3 of 4 NI tests with Holm p < 0.05; otherwise INCONCLUSIVE.
- **Secondary families** (Holm within each; flagged secondary): anchor-far stratum (both test images > 0.03 from every typed anchor; far AUC, far accuracy); HTR (HTR accuracy, HTR log-loss); time-blocked (AUC, accuracy); cold start (AUC, accuracy). A candidate that is BETTER on `main` but has a non-positive far gain in any cell is reported as "gain only near anchors".
- Descriptive: per-type gains, anchor strata (≤ 0.001, 0.001–0.03, > 0.03, > 0.01), controls.
- **Headline rule:** a candidate that is BETTER in both splits (≥ 3 of 4 Holm) on the standard endpoint. If none, the answer to the question is the per-candidate verdict table.
Expectation stated in advance (from development): `fs_lam10`, `hy_`, `ds_` are WORSE or INCONCLUSIVE; `rt_c2_…` is the only candidate with a positive development signal (AUC ≈ +0.005…+0.008, accuracy ≈ +0.0045 — about the Holm-level detectability limit of 35 seeds for accuracy, so BETTER may fail on accuracy), and its development estimate is optimistic (the routed type was chosen post hoc on those seeds).

## Runs
```
bash scripts_sp/run_sp.sh conf1 lap3_k10_st4,aw8_all_st2lr0.5,rt_c2_lam10_rho0.001,fs_lam10,hy_lamr10_rhor0.1,ds_mu0.03_lam10,rt_c2_lam10_rho0.001_sh1,rt_c2_lam0,fs_lam10_sh1,fs_lam0,kn,ls,hy_lamr10_rhor0.1_sh1 11230-11264 3 A,B groups
bash scripts_sp/run_sp.sh conf1_ctl lap3_k10_st4,ds_mu0.03_lam10_sh1 11230-11264 1 A,B groups
bash scripts_sp/run_sp.sh conf1_blocked lap3_k10_st4,aw8_all_st2lr0.5,rt_c2_lam10_rho0.001,fs_lam10,hy_lamr10_rhor0.1,ds_mu0.03_lam10,rt_c2_lam10_rho0.001_sh1,rt_c2_lam0,kn,ls 11230-11264 2 blocked groups
bash scripts_sp/run_sp.sh conf1_cold <same variants> 11230-11264 1 A,B random
python3 sp_confirm.py --spec SP1_spec.json --dirs conf1 conf1_ctl --blocked conf1_blocked --cold conf1_cold
```
Note on compute (added at ≈ 06:05, after the main run had started and before the secondary runs started): a confirmation cell costs ≈ 50 s, so the blocked and cold-start runs omit the shuffled / no-penalty controls of the three non-headline candidates (`fs_lam10_sh1`, `fs_lam0`, `hy_…_sh1`, `ds_…_sh1`); the headline route keeps its shuffled and no-penalty controls everywhere. Confirmation 2 is run only if time allows.

Limits: same 168 labelled groups (re-split, not new data); ≈ 56–70 decisive test judgments per (seed, split), of which only ≈ 10 are HTR; the anchor-far stratum keeps ≈ 15 % of the judgments; the labelled data are 284 frames of three growth runs; the routed type and its scale were chosen on development seeds; winner's-curse: 4 candidates chosen among ≈ 45 learners tried.

## Amendment SP1b (written 2026-10-14 ≈ 14:40, after confirmation 1's standard endpoint had been analysed and before any confirmation-2 seed was run)
Confirmation 1 (standard endpoint) gave: `rt_c2_lam10_rho0.001` BETTER (3 of 4 Holm tests significant), `fs_lam10` WORSE, `hy_lamr10_rhor0.1` WORSE, `ds_mu0.03_lam10` NON-INFERIOR (generated tables in `RESULTS_score_propagation.md`). Confirmation 2 (independent seeds 11265–11299, standard splits A and B, 3 label sets per budget, run once) therefore replicates **only the headline candidate**: variants `lap3_k10_st4`, `aw8_all_st2lr0.5`, `rt_c2_lam10_rho0.001`, `rt_c2_lam10_rho0.001_sh1`, `rt_c2_lam0`; the main family is the 4 tests of `rt_c2_lam10_rho0.001` (splits × {AUC, accuracy}, Holm over 4, same BETTER rule: all four gains positive and ≥ 3 of 4 Holm p < 0.05), the far / HTR families are its 4 tests each; the other three candidates are not re-run (their classification in confirmation 1 was decisive or null). Command: `bash scripts_sp/run_sp.sh conf2 lap3_k10_st4,aw8_all_st2lr0.5,rt_c2_lam10_rho0.001,rt_c2_lam10_rho0.001_sh1,rt_c2_lam0 11265-11299 3 A,B groups`; analysis `sp_confirm.py --spec SP1b_spec.json --dirs conf2`.

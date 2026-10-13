# Paper package: one honest, reproducible results package for the learner / graph / selection line

Branch `claude/paper-package-20261013` (cut from `claude/redteam-20261013`; result trees and documents of `claude/graph-mechanism-20261013` and `claude/graph-endpoint-1013` were copied in unchanged). **Nothing was re-run**: every table and figure is generated from the stored result files by the scripts in `graph_exploration/PAPER_PACKAGE/` (`make all`); every number in the prose is checked against the generated tables by `check_prose.py`. `code/ACADEMIC_REPORT_DRAFT.md` is untouched. This document is assembled by `build_docs.py`; edit `prose/*.src.md`, never this file.

## 0. How to read this
- **Status words** (never stronger than the frozen rule / the data): *PRE-REGISTERED, CONFIRMED* = the success rule written before the confirmation seeds existed was met in the primary block (further blocks: replicated / partly / not replicated); *PRE-REGISTERED, NOT CONFIRMED*; *DESCRIPTIVE* = computed on confirmation seeds but not itself a frozen test; *DEVELOPMENT OBSERVATION* = development seeds only (selected on the same seeds); *AUDIT* = red-team development audit; *DOCUMENTED ONLY* = earlier protocol, numbers not regenerated here.
- **Gains** are paired over seeds on identical label sets (mean over budgets 10/20/40/60 and 3 label draws); intervals are 95%⟦-⟧ bootstrap over seeds; splits A / B are the two held-out designs. Same 168 labelled groups re-split in every seed: intervals and p-values are optimistic (section 4).
- **Pre-registration order** was verified from git history: for 22⟦prov.verified⟧ of 25⟦prov.total⟧ (study, result directory) pairs the frozen pre-registration file was committed before the first commit of the confirmation result directory (`tables/prereg_order.md`); the remaining 3⟦prov.not_verifiable⟧ are red-team confirmation directories that did not exist in the git history when the package was built. Commit time is not file-creation time: the check shows ordering of commits, not that no cell existed earlier (the studies state they did not).

## 1. Claims table (`tables/claims_onepage.md` = one-page view; `tables/claims_table.md` = full table with intervals, seeds and evidence; CSV: `tables/claims.csv`)
### 1a. One-page view
{{INCLUDE:tables/claims_onepage.md}}

### 1b. Full table
{{INCLUDE:tables/claims_table.md}}

### Documented-only claims (earlier protocol; numbers not regenerated)
{{INCLUDE:tables/claims_appendix.md}}

### What to put in a paper, in three tiers
- **Safe headline claims** (pre-registered, replicated, and not an anchor-copy artefact): anchor weighting (L1, L2; anchor-far AUC +0.044⟦st.dev.aw8_all.baseline.A.far>0.03.auc.gain⟧ / +0.035⟦st.dev.aw8_all.baseline.B.far>0.03.auc.gain⟧ in the red-team development block); selection rules are absorbed by a tuned learner (S1 vs S2/S3).
- **Claims to hedge**: the Laplacian learner and the long schedule (L3 to L5) are confirmed on the standard endpoint, but the red-team audit locates most of the gain in anchor-near judgments (while a time-blocked split keeps it: R3); the graph's own share beyond generic regularisation is small and unconfirmed (D2); HTR gains are fragile (L13, S4 to S6).
- **Do not claim**: that more unlabelled images keep helping (G4), that beating a shuffled graph proves the graph structure matters (L8, G3), a graph advantage on the paper's own type-accuracy endpoint (G6), robustness to unseen growth runs or samples (only three runs exist; harsh-split tests use fixed small test sets, G5).

## 2. Draft abstract-level summary (English; also `PAPER_PACKAGE/ABSTRACT_EN.md`)
{{INCLUDE:build/abstract_EN.md}}

## 3. Figures (script-generated, one script per figure in `PAPER_PACKAGE/figures/`; PNG and PDF)
- **Fig. 1a/1b** `fig1a_learning_curves`, `fig1b_gain_over_aw8_all` — learning curves (AUC, accuracy, log-loss vs budget; 95%⟦-⟧ bootstrap bands over 70⟦-⟧ seeds per block) for the default head, aw8, aw8_all, aw8_all_st2lr0.5, lap3_k10, lap3_k10_st4. The default head, aw8 and aw8_all were measured in one seed block, the others in another (aw8_all in both); compare inside a block (1b). In split A the best learner with 10⟦-⟧ added judgments (AUC 0.933⟦cv.S.lap3_k10_st4.A.auc.b10⟧) is above aw8_all with 60⟦-⟧ added judgments (0.916⟦cv.S.aw8_all.A.auc.b60⟧).
- **Fig. 2a/2b** `fig2a_per_type_accuracy`, `fig2b_per_type_gain` — per-type panels, HTR boxed (tiny per-type test samples).
- **Fig. 3** `fig3_ladder` — learner ablation ladder; the four steps add up (indicatively, different seed blocks) to +0.044⟦ld.SUM.A.auc.gain⟧ / +0.050⟦ld.SUM.B.auc.gain⟧ AUC over the default head.
  Pooled over the two pre-registered blocks (70⟦ld.POOL.C2.A.auc.n⟧ seeds) the final learner beats the tuned graph-free control by AUC +0.018⟦ld.POOL.C2.A.auc.gain⟧ [+0.015⟦ld.POOL.C2.A.auc.lo⟧, +0.021⟦ld.POOL.C2.A.auc.hi⟧] / +0.016⟦ld.POOL.C2.B.auc.gain⟧ [+0.013⟦ld.POOL.C2.B.auc.lo⟧, +0.020⟦ld.POOL.C2.B.auc.hi⟧] and accuracy +0.021⟦ld.POOL.C2.A.acc.gain⟧ / +0.019⟦ld.POOL.C2.B.acc.gain⟧ (A / B; `tables/ladder.md`, rows `POOL-*`).
- **Fig. 4** `fig4_selection` — selection rules vs random as the learner gets stronger.
- **Fig. 5** `fig5_dose_response` — dose-response of the graph: noise, random vs local rewiring, share of unlabelled frames; shuffled graph off scale.
- **Fig. 6 / 6b** `fig6_anchor_strata`, `fig6b_anchor_fractions` — gain by anchor proximity (anchor-near / anchor-far) and the share of judgments per stratum.

{{INCLUDE:build/assumptions_limits.md}}

## 5. Integrity checks that are part of the build
- `pp_confirmations.py` recomputes every pre-registered test with each study's own test code and frozen spec; `tests/test_crosscheck.py` compares a sample with the numbers printed in the studies' own result documents.
- `pp_strata.py` asserts equality with the red team's `rt_tables.auc_by_stratum`.
- `pp_seedblocks.py` asserts that the 21⟦seeds.n_blocks⟧ declared seed blocks are disjoint and that every confirmation directory contains only seeds of its declared block (`tables/seed_blocks.md`, `tables/inventory.md`).
- `pp_provenance.py` verifies pre-registration order from git history (`tables/prereg_order.md`); `check_prose.py` fails the build on any prose number that does not match a generated table.

## 6. Reproduce
`cd code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/PAPER_PACKAGE && make all` (about 3⟦-⟧ minutes on one core; needs numpy, pandas, scipy, scikit-learn, matplotlib; no torch). See `README.md` there for the directory / seed block / branch map and `HANDOFF_paper_package.md` for what to do next.

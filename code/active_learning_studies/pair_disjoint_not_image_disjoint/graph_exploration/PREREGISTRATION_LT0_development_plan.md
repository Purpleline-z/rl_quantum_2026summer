# Development plan LT0 (latent-type study) — written before any development result was read

Branch `claude/latent-type-1014`, seeds: development **10700–10719**, confirmation 1 = 10730–10764, confirmation 2 = 10765–10799 (`nd_core.LT_*`, assertions against every earlier block). Nothing from 10730 on has been run.

## Model (code: `new_directions/lt_model.py`, `lt_learner.py`)
Latent type z_i ∈ {(1x1), c(6x2), RT13, HTR, none} per image. Judgment outcome ∈ {a wins, b wins, tie, not_apply} given s_a = 1[z_a = k], s_b = 1[z_b = k]: confusion table T[s_a, s_b, o] with swap symmetry, shared across types, Dirichlet pseudo-counts that encode the verbal outcome definitions only. Prior over z_i = label-spreading posterior of the typed ideal images (kNN graph, α = 0.9) with floor 0.05 and "none" mass ρ = 0.15. Several types of the same image pair form one pairwise factor. E-step: loopy BP; M-step: expected counts. EM runs inside every fit on the labelled judgments of that fit only (no test, no validation information).
Uses of the posterior inside `lap3_k10_st4` (everything else unchanged; exact equivalence with all terms off is unit-tested): `pa<w>` soft pseudo-anchors, `lik` answerability-aware weighting of the BT terms, `aux<w>` auxiliary cross-entropy to the posterior.
Controls: `prior` (no outcome likelihood = graph prior only; same downstream use), `random` (EM posteriors permuted among the images), `shufprior` (prior built on the shuffled graph). Baseline `lap3_k10_st4`; tuned graph-free control `aw8_all_st2lr0.5`.

## Development screens (seeds 10700–10719; every variant is logged in `PROGRESS_LOG_latent_type.md`)
- D1: `pa0.25_em`, `pa1_em`, `pa1_prior`, `lik_em`, `lik_prior`, `aux0.5_em`, `aux0.5_prior` vs `lap3_k10_st4` and `aw8_all_st2lr0.5`; splits A and B, budgets 10/20/40/60, 1 random label set per budget. Reported: standard endpoint and anchor-far stratum (both images > 0.01 cosine distance from every typed ideal image).
- D2 (leave-anchors-out diagnostic, `lt_diag.py`): hide the labels of 25% of the 116 non-test typed ideal images; type recovery by graph prior alone vs prior + outcome posteriors (+ random / shuffled-prior controls); near vs far from the judgment images.
- D3 (follow-ups chosen after D1/D2, ≤ 8 further variants: ρ, pseudo-count strength, per-type table, weights, combinations).
- Time-blocked split run only for the frozen candidates (also in confirmation).
Freeze (≤ 4 candidates + controls + success rule) is committed in `PREREGISTRATION_LT1.md` before any seed ≥ 10730 is run.

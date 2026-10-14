# Pre-registration AF1 (anchor-far study) — frozen before any confirmation seed was run

Written 2026-10-14 (UTC ~05:05), committed before seeds 10830–10899 were used. Branch `claude/anchor-far-20261014`. Scripts: `af_learner.py`, `af_tables.py`, `af_confirm.py`, `af_power.py` (all committed before this file).

## Where we stand (development, seeds 10800–10811, splits A and B, budgets 20/40, 2 draws)
Round 1 (14 learners) and round 2 (11 learners) screened the six requested families; tables `dev_round1_table.md`, `dev_round2_table.md`; time-blocked screen on the same seeds in `results/rt/af_b1`.
**No variant improved the anchor-far (both images > 0.01 cosine distance from every typed anchor) AUC and accuracy in both splits** over the baseline `lap3_k10_st4` (= `af_base`).  Interpolation-consistency anchor–node/anchor–near-node, per-run centring and the shuffled pseudo-anchor control were clearly harmful (−0.02 … −0.2 AUC); the rest are within about ±0.01.  The expectation entering the confirmation is therefore a **null**; the confirmation is run to make that statement at 35 seeds with controls instead of at 12.

## Frozen candidates (≤ 4 incl. controls count as separate learners)
| candidate | mechanism | control |
|---|---|---|
| `af_pseudoW0.3` | graph-propagated typed posteriors of the references as soft pseudo-anchors, confidence max q·exp(−hop/4), ≤ 8 hops, weight 0.3, anchor ranking loss | `af_pseudoW0.3_shuf` (graph node identities permuted) |
| `af_nbra1` | anchor-neighbourhood smoothing: 30 nearest nodes of each typed anchor, edge weights exp(−rank/10), λ_a = 1 | `af_nbra1_shuf` |
| `af_mixaa0.3` | mixup between anchors of different types with monotone ranking along the mix path (0.8 vs 0.3), weight 0.3 | `af_mixaa0.3_perm` (anchor classes permuted) |
Reference: `af_base` (bit-identical to `lap3_k10_st4`, unit test).  Comparator reported, not tested: `aw8_all_st2lr0.5` (tuned graph-free).  Learner specs: `af_spec_AF1.json`; hyper-parameters are exactly the registered learner names in `af_learner.py` — no further tuning.

## Protocol
Confirmation 1: seeds 10830–10864 (35), splits A and B, budgets 20 and 40, 2 draws, identical random label sets for all learners (`nd_core.random_labelled`); per-judgment predictions (`rt_a1.py`), anchor distances (`rt_a1b.py`), time-blocked split (`rt_a3.py`, 100-frame blocks, 25-frame buffer).  Confirmation 2 (10865–10899) only if confirmation 1 shows a candidate that passes the rule below on at least the AUC side; otherwise not run (nothing to confirm).  Each seed block is run once.

## Tests (`af_confirm.py`)
Per seed: mean over budgets and draws of the paired gain (candidate − `af_base`); two-sided Wilcoxon signed-rank over seeds; paired bootstrap CI over seeds.
* **Primary family** (Holm over 12 tests): 3 candidates × {A, B} × {AUC, accuracy} on the **anchor-far stratum** (both images > 0.01 from every typed anchor: references, utility images, outer-test ideal images).
* **Blocked family** (Holm over 6 tests, reported with equal weight): 3 candidates × {AUC, accuracy} on the time-blocked split.
* **Standard endpoint** (all held-out judgments, A and B, AUC and accuracy): not tested, only the non-inferiority condition below.
* **Cluster-aware add-on** (descriptive, outside the Holm decision): two-way bootstrap over seeds and pair groups (`af_power.py`) for every primary test.  Reason: all seeds of a split share nearly the same ~50–65 held-out judgments (about a third in the far stratum), so seed-level p-values overstate precision (red-team data: SE 2–3× larger when pair groups are resampled; MDE80 ≈ 0.03–0.04 AUC for the far stratum).
* Dose-response: AUC and accuracy gain by distance to the nearest typed anchor (bins ≤0.001, 0.001–0.01, 0.01–0.03, 0.03–0.1, >0.1) and the share of judgments per bin — descriptive.

## Success rule (per candidate; from the task statement)
(1) the anchor-far AUC and accuracy gains are positive in both splits (4/4 positive); (2) ≥ 3 of these 4 tests have Holm-adjusted p < 0.05; (3) the standard-endpoint AUC gain is ≥ −0.002 in A and in B; (4) for a graph/anchor-based claim the candidate's far gain exceeds that of its control in all 4 cells.  A candidate satisfying 1–4 is "confirmed"; anything else is "not confirmed".  The time-blocked gains are reported as the second primary endpoint; a candidate is called "supported on blocked too" only if both blocked gains are positive and ≥ 1 is Holm-significant.  Wording rule: no candidate is described as an improvement unless it is confirmed; observations from strata or dose-response are exploratory.

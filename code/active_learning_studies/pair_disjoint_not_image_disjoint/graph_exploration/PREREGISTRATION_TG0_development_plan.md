# TG0 — development plan of the temporal-graph study (written 2026-10-13 16:20 UTC, before any development result was inspected)

Branch `claude/temporal-graph-20261013` (cut from `claude/redteam-20261013`). Seeds: development 9200–9219 (used for design only; 9220–9249 spare for further development), confirmation 9250–9284 (untouched until the freeze commit `PREREGISTRATION_TG1.md`). Common rules: `prompts/COMMON_RULES.md`.

**Question.** Does adding temporal edges (adjacent frames of one growth run) to the Laplacian learner improve the held-out preference endpoint over the incumbent `lap3_k10_st4` (= `aw8_all` + Laplacian λ = 3 on the k = 10 cosine kNN graph of unlabelled images + 4x head steps)?

**Not repeated from the parallel branches.** `graph-mechanism` (graph quality dose-response, constructions of the *feature* graph, RT13-off) and `graph-endpoint` (classification endpoint, node-set size, harsh splits incl. a "forward in time" split) did not test temporal edges inside the learner. Earlier negative temporal attempts (contrastive adapter, filename metadata fusion) used different mechanisms.

**Variants (development; baseline `lap3_k10_st4`; all with mu = 3, EDGE_BATCH 256 per step, the temporal term sampled from its own random stream so the kNN stream of the baseline is unchanged; exact equivalence with the baseline when no temporal term, unit-tested).**
1. `tg_u1`, `tg_u2`, `tg_u5` — union: kNN term + chain edges between frames of the same run with index gap <= 1 / 2 / 5 (one family, three windows).
2. `tg_only2` — temporal-only graph: no kNN term at all, chain +-2.
3. `tg_inter` — intersection: trajectory–trajectory kNN edges kept only if same run and index gap <= 25 (edges touching ideal images kept).
4. `tg_dec5` — chain +-5 with weights exp(-gap/2) * exp(-|dT|/30).
5. `tg_tbin` — edges between 10 random frames (any run) of the same 20 C temperature bin per frame.
6. `tg_curv` — second-difference penalty (s(i-1) - 2 s(i) + s(i+1))^2 along each run, in addition to the kNN term.
7. `tg_unl2` — chain +-2 only between frames in no labelled judgment of the current fit.
8. `tg_anc2` — kNN edges only where an ideal image is an endpoint, plus chain +-2 (is the trajectory–trajectory kNN part replaceable by time?).
9. `tg_novel5` — only those chain +-5 edges that are NOT already kNN edges (the information beyond the feature graph).
(10 variant types counting the chain windows as one; 11 learners + baseline.)

**Controls.** `_shuf`: temporal-edge endpoints permuted within each run (degree preserved) — the control asked for; `_lshuf`: permuted locally (displacement <= 25 positions); `_krand`: the same number of random kNN edges (generic similarity without time). `_strict`: temporal edges dropped when an endpoint is within 25 *frame numbers* (same run) of any held-out image.

**Endpoints.** Standard endpoint (AUC, accuracy, log-loss; HTR accuracy), **anchor-far** (both test images > 0.03 cosine distance from every typed ideal image; `rt_a1b.py` definition; ~30 % of test judgments), anchor-near (complement), and the red-team **time-blocked split** (`rt_blocked.py`; secondary). A gain that exists only in the anchor-near stratum is reported as such.

**Screening protocol.** Stage 1: 12 learners x seeds 9200–9209 x splits A, B x 4 budgets x 1 draw (random label sets identical across learners). Stage 2 (seeds 9210–9219 and controls) for what looks non-null. Decision for the freeze: <= 4 candidates + controls, chosen on paired AUC/accuracy gains over the baseline, requiring a better-than-shuffle control in development; no confirmation seed is run before the freeze commit.

**Expectation stated in advance.** 84–94 % of the chain edges are already kNN edges (measured, see results), so the temporal term mostly re-weights existing edges; the likely outcome is a null with harm from variants that add feature-incoherent edges (`tg_tbin`, `_shuf`). A positive result would have to come from the non-kNN edges (`tg_novel5`), the curvature penalty, or from re-weighting.

# Autonomous run (started 2026-10-09, user asleep, at least 8 hours): make a graph-aware acquisition rule significantly better than Random

Source of truth for this run. Update it at every phase; a later context must be able to continue from this file alone.
Branch `claude/graph-active-selection`. Language to the user: Chinese, plain, hedged, no overclaiming.

## Prompt (to myself)
Goal: find a graph-aware (image-similarity graph, type-aware, possibly GCN-based) acquisition rule that is significantly better than Random on the held-out preference endpoint, in the judgment-unit
framework (query unit = one (pair, type) judgment, budgets 10/20/40/60 judgments, splits A and B, single-shot and sequential rounds of 10). Read the literature on how others do it; do not guess.
Protocol is FIXED and must not change: endpoint, splits, head schedule (`results/pair_endpoint_study/schedule.json`), seeds, budgets, Random = mean of 5 draws, Wilcoxon over seeds + Holm.
Only selectors (new rules in `graph_strategies.py`/new modules) may change. A selector may see only: candidate judgments, revealed judgments, the model, cached features of their images, and the typed
reference (ideal) images that are already training anchors in every strategy (`exp.references`; never `exp.test_images`/`utility_images`, never validation/test group images).
Honesty rules: report failure if it happens; no tuning on confirmation seeds; every number in reports generated from cells; list all rules tried (including failures); state multiple-comparison caveats.

## Selection-bias protocol (decided before any new method was run)
- DEV seeds 400-409 (both splits, both conditions): all design iteration and tuning happens here only.
- CONFIRM seeds 410-429 plus 42, 79, 123, 202, 303 (25 seeds): used once, after freezing at most 4 candidate rules and writing them into this file. Holm over those candidates (+ their controls reported separately).
- Caveat: the first-generation graph rules (centrality, bridge, graph_core_set; report §5.14) were already evaluated on all 35 seeds, so what they showed (weak split-A hint for centrality) informs design; their confirm-seed results are not independent evidence for new rules.
- Success criterion (fixed now): a frozen candidate has Holm p < 0.05 for log-loss gain over Random in at least two of the four blocks (split x condition), including both splits, and a non-negative AUC gain. Otherwise report "not achieved" with the numbers.
- Power note: per-seed sd of the gain is 0.09-0.14 log-loss, so with 25 seeds an effect below about 0.04-0.05 is unlikely to be significant.

## To-do (status: [ ] open, [~] in progress, [x] done)
- [ ] P0 literature: how graph-based active learning and preference active learning do it (Sequential GCN details, GRAIN/FeatProp, pairwise-preference AL, low-budget AL); write findings to `notes/graph_literature_round.md`.
- [ ] P1 implement the literature graph baselines faithfully on the image graph: UncertainGCN and CoreGCN (Caramalau et al. 2021), with typed reference nodes.
- [ ] P2 type-aware graph signals: label propagation of the reference-image type posterior over the kNN graph; per-judgment features (type-t posterior gap of the two images, graph distance to revealed judgments of the same type).
- [ ] P3 combined rules (coverage x predicted-decisive x own-head uncertainty on the graph); iterate on DEV seeds, log each round below.
- [ ] P4 freeze <= 4 candidates here, then run CONFIRM seeds, aggregate, Holm.
- [ ] P5 report section, tests, commit/push (restore any manifests rewritten by old tests before committing; add only specific paths).

## Log (append)

### 2026-10-09 (night) setup
- User clarified: large GPU tasks are left for the user (list in `GPU_TASKS_FOR_USER.md`); I do small CPU experiments and literature reading.
- Literature (notes_graph_literature_round.md): Sequential GCN verified from the authors' code (dense normalised cosine adjacency, labelled-vs-pool BCE, k-centre on first hidden layer). arxiv/ar5iv/CVF are blocked, GitHub raw works.
- Data-level diagnostic (all 521 rows, 5-fold by group, 10 repeats; design-informing, uses labels of all groups): label propagation of reference-image types over the kNN graph (k=10, alpha=.9) gives 8 compact features per judgment; AUC for decisive 0.828 (type only 0.599, type+cosine 0.706, 1024-d mean+|a-b| 0.795), tie 0.879 (1024-d 0.855), not_apply 0.897 (1024-d 0.752). k=5 was worse (0.727 decisive); only k in {5,10} and alpha .9 were tried. So `P(decisive)` from graph features is a genuinely graph-based, type-aware signal that can be fitted with ~30 revealed judgments.
- Implemented `graph_typed.py`: ImageGraph (nodes = candidate + revealed + typed reference images; posterior q), `decisive_probability`, rules typed_decisive_{uncertainty,bald,coverage} (+ shuffled-posterior controls), coregcn, uncertaingcn. `judgment_unit_study.make_candidates` now also puts the typed reference images into the embedding cache (extra cache keys only; no other strategy reads them).
- DEV reference bars (embedding-based, no graph) running on seeds 400-409: core_set_relation, laplace_bald, bald_decisive, fisher_dopt, typiclust_pairs, dpp_pairs.

### Round 1 (DEV seeds 400-409; cell = mean log-loss gain over Random; blocks A-single, A-seq, B-single, B-seq)
- typed_decisive_coverage +0.095 / +0.109 / +0.024 / +0.105 (its shuffled-posterior control +0.042 / +0.008 / +0.005 / +0.079): best so far; real minus shuffled +0.053 / +0.101 / +0.019 / +0.026.
- typed_decisive_bald +0.100 / +0.068 / -0.021 / +0.078 (shuffled +0.066 / -0.001 / -0.055 / +0.018); typed_decisive_uncertainty +0.082 / +0.099 / -0.027 / +0.017 but its shuffled control is as good in A (+0.093 / +0.076): in A the gain is from the type-dependent decisive prior, not the graph.
- Literature baselines on the graph: coregcn +0.010 / +0.011 / -0.006 / +0.034, uncertaingcn +0.050 / +0.036 / -0.022 / +0.014 (not better than the embedding baselines core_set_relation +0.041 / +0.058 / -0.036 / +0.062; bald_decisive -0.003 / +0.001 / +0.007 / +0.078).
- AUC gains are small (<= +0.024). Split B single-shot is hard for every rule (most are negative).
- Next (round 2): decisive-weighted sampling (gamma 1/2/4), type-only controls (isolate what the graph adds to the decisive predictor), coverage with p^2 and with an uncertainty factor.

### Round 2 (DEV 400-409): log-loss gain over Random, blocks A-single, A-seq, B-single, B-seq
- typed_decisive_coverage_unc (coverage x P(decisive) x (0.25 + own-head uncertainty)): +0.108 / +0.138 / +0.029 / +0.091 (AUC +0.019 / +0.022 / +0.005 / +0.020). Best.
- typed_decisive_coverage_g2 (P(decisive)^2): +0.132 / +0.120 / +0.005 / +0.065. typed_decisive_coverage (round 1): +0.095 / +0.109 / +0.024 / +0.105.
- Control typeonly (decisive predictor from the type one-hot only; same coverage): +0.082 / +0.101 / +0.022 / +0.056 with AUC about 0 (+0.001 / 0.000 / +0.001 / +0.009). So most of the log-loss gain comes from type-aware decisive weighting; the graph features (label-propagated type posterior) add about +0.013 / +0.008 / +0.002 / +0.049 log-loss and +0.012 / +0.010 / -0.003 / +0.006 AUC for the plain coverage rule.
- Decisive-weighted random sampling (gamma 1/2/4) is not good (B-single -0.06 to -0.10): coverage is needed.
- Round 3: variants around coverage_unc (gamma 2, uncertainty power 2, q-coordinates in the coverage space) plus its type-only and shuffled controls. Then freeze <= 4 candidates.

### Round 3 (DEV 400-409): log-loss gain over Random, blocks A-single, A-seq, B-single, B-seq
- coverage_unc +0.108 / +0.138 / +0.029 / +0.091 (AUC +0.019 / +0.022 / +0.005 / +0.020); its type-only control +0.119 / +0.056 / -0.017 / +0.056 (AUC +0.010 / -0.005 / -0.006 / +0.001); its shuffled-posterior control +0.103 / +0.096 / -0.043 / +0.022 (AUC +0.008 / +0.009 / -0.006 / +0.003).
  So on DEV the real graph posterior beats both controls in A-seq, B-single and B-seq on log-loss and in all four blocks on AUC; in A-single the type-only control is as good.
- Variants not better than coverage_unc: unc_u2 (uncertainty squared) +0.116 / +0.145 / -0.009 / +0.070; unc_g2 +0.098 / +0.109 / -0.003 / +0.067; unc_q (type-posterior coordinates in the coverage space) +0.077 / +0.088 / +0.009 / +0.034.

### FROZEN CANDIDATES (written before any confirmation seed was run; DEV selection only)
1. `typed_decisive_coverage_unc`   (primary)
2. `typed_decisive_coverage`       (plain coverage x P(decisive); strongest in B-sequential on DEV)
3. `typed_decisive_bald`           (Laplace-BALD x P(decisive); information-based variant)
Controls reported alongside, not candidates: `typed_decisive_coverage_unc_typeonly`, `typed_decisive_coverage_unc_shuffled`. Embedding baselines on the same confirm cells (second wave): core_set_relation, bald_decisive, typiclust_pairs, laplace_bald, uncertainty.
Test: seed-level Wilcoxon on the log-loss gain over Random (mean over budgets 10/20/40/60), Holm over the 3 candidates within each of the four blocks; AUC reported. Success criterion as fixed at the top of this file.
CONFIRM seeds: 410-429 and 42, 79, 123, 202, 303 (25 seeds). No changes to the candidates after this point.

### CONFIRM results (25 unseen seeds; frozen candidates; Holm over the 3 candidates per block). Full output: results/judgment_unit_study/confirm_report.txt (regenerate with `python3 aggregate_confirm.py`)
**The pre-registered success criterion is NOT met.** No candidate has Holm p < 0.05 in both splits.
- typed_decisive_coverage (plain coverage x P(decisive)): split B significant in both conditions (single-shot +0.062 log-loss, Holm p 0.014, AUC +0.013; sequential +0.083, Holm p 0.002, AUC +0.019) but not in split A (-0.007 and -0.010, Holm p 1.0).
- typed_decisive_coverage_unc (primary): A -0.002 / -0.008, B +0.002 / +0.050 (Holm 0.055): not significant anywhere. typed_decisive_bald: A +0.007 / -0.006, B +0.019 / +0.049 (Holm 0.055).
- Controls: the shuffled-posterior control of coverage_unc is as good as or better than the real rule in B-sequential (+0.081 vs +0.050) and in B-single (+0.031 vs +0.002); the type-only control is +0.026 / +0.030 in A and -0.005 / +0.040 in B. The graph type-posterior therefore does NOT explain the B gain; coverage in graph-propagated pair space does, and coverage rules without any graph (core_set_relation: A +0.042 / +0.050, B-seq +0.075) do as well.
- The development gains (+0.1 to +0.14 in split A) did not replicate (A about 0 on 25 unseen seeds). Reason checked: on dev seeds 400-409 Random's log-loss was unusually poor (0.648 vs 0.548 on confirm seeds) while the candidate's own log-loss was similar (0.510 vs 0.554); with 10 dev seeds and about 20 variants tried, the dev choice was optimistic (selection on seeds where Random happened to do badly).
- Wave 2 (embedding baselines on the same confirm cells) running; numbers above for those rows are partial in split B until it finishes (n=20 of 25).

### Round 4: REPLICATION on fresh seeds 430-459 (30 seeds, split B only, both conditions). Pre-registered before running (2026-10-09 19:45 UTC).
Motivation: `typed_decisive_coverage` was significant in split B in the confirmation (+0.062 single-shot, Holm 0.014; +0.083 sequential, Holm 0.002) but its own controls were never run, and the controls of the primary rule showed that the type posterior is not responsible. No tuning: the rules are exactly the existing `typed_decisive_coverage`, `typed_decisive_coverage_typeonly` (decisive predictor from the type one-hot only), `typed_decisive_coverage_shuffled` (type posterior shuffled over images), and the embedding baseline `core_set_relation`; Random = 5 draws.
- H1 (primary): typed_decisive_coverage has a positive log-loss gain over Random in split B; seed-level Wilcoxon, Holm over the two conditions.
- H2 (does the graph help?): real minus shuffled and real minus type-only, paired over seeds, reported with raw p (4 comparisons: 2 controls x 2 conditions; not corrected, say so).
- Not tested: split A (null in the confirmation).
Interpretation rule decided now: if H2 differences are not clearly positive, the conclusion is "coverage in graph-propagated space helps in split B, the type posterior / graph structure is not shown to be responsible".

### Round 4 results (seeds 430-459, split B; results/judgment_unit_study/replication_report.txt)
typed_decisive_coverage: single-shot +0.042 log-loss (raw p 0.029, Holm over the two conditions 0.059, AUC +0.008), sequential +0.030 (Holm 0.39). Controls as good: type-only +0.038 / +0.038, shuffled posterior +0.060 / +0.034; paired rule minus type-only +0.004 / -0.008, minus shuffled -0.018 / -0.003 (all n.s.). Core-set baseline -0.015 / -0.019 (its confirmation gain of +0.063 did not replicate); rule minus Core-set +0.057 (raw p 0.058) / +0.049 (0.12).
Verdict: H1 not significant after Holm (single-shot borderline); H2 not supported (graph posterior adds nothing over the type-only decisive predictor). Honest summary: type-aware decisive weighting of a coverage rule gives a small gain (about 0.03-0.06 log-loss) in split B; the graph is not responsible; split A null.
Report: section 5.15 now contains the confirmation, the verdict and the replication (generated by make_confirm_section.py --insert); abstract and conclusion were NOT changed.

### ---- Second conversation (branch claude/gallant-knuth-3ikn96), merged here ----
### Round 3 (DEV 400-409, finished 2026-10-09 in a fresh container; feature cache rebuilt, `launch_round.sh` path bug fixed)
Log-loss gain over Random, blocks A-single / A-seq / B-single / B-seq (AUC gain in the second line):
- typed_decisive_coverage_unc (reference): +0.108 / +0.138 / +0.029 / +0.091; AUC +0.019 / +0.022 / +0.005 / +0.020.
- ..._u2 (uncertainty^2): +0.116 / +0.145 / -0.009 / +0.070; AUC +0.021 / +0.024 / +0.002 / +0.014. Better in A, worse in B.
- ..._g2 (P(decisive)^2): +0.098 / +0.109 / -0.003 / +0.067. ..._q (coverage in q-coordinates): +0.077 / +0.088 / +0.009 / +0.034. Both worse than the reference.
- Controls of coverage_unc: typeonly +0.119 / +0.056 / -0.017 / +0.056 (AUC +0.010 / -0.005 / -0.006 / +0.001); shuffled-posterior +0.103 / +0.096 / -0.043 / +0.022 (AUC +0.008 / +0.009 / -0.006 / +0.003).
  Real minus shuffled: +0.005 / +0.042 / +0.072 / +0.069 log-loss, AUC +0.011 / +0.013 / +0.011 / +0.017. In split A single-shot the type-dependent decisive prior alone (typeonly, shuffled) already gives +0.10 to +0.12; the graph signal helps mostly in sequential and in B.
- No variant beats the reference overall (10 seeds; most differences are within noise). Stop tuning.

### FROZEN CANDIDATES (written before any confirmation seed was run)
1. typed_decisive_coverage_unc  (best DEV mean)
2. typed_decisive_coverage      (simplest; best B-seq on DEV)
3. typed_decisive_bald          (second family: BALD instead of own-head uncertainty)
Controls (reported, not candidates, not in the Holm family): typed_decisive_coverage_unc_typeonly, typed_decisive_coverage_unc_shuffled. Baseline = Random (cells exist for all 35 seeds).
Holm family = these 3 candidates x 4 blocks (per metric). CONFIRM seeds 410-429 + 42, 79, 123, 202, 303, run once. Success criterion as above.

### CONFIRMATION RESULT (25 seeds: 410-429, 42, 79, 123, 202, 303; run once; `aggregate_confirm.py`; full tables in report §5.15)
Success criterion NOT MET. Log-loss gain over Random (A-single / A-seq / B-single / B-seq):
- coverage_unc -0.002 / -0.008 / +0.002 / +0.050 (Holm 1 / 1 / 1 / .28); coverage -0.007 / -0.010 / +0.062 / +0.083 (Holm 1 / 1 / .051 / .008); bald +0.007 / -0.006 / +0.019 / +0.049 (none significant).
- Controls of coverage_unc: typeonly +0.026 / +0.030 / -0.005 / +0.040; shuffled +0.040 / +0.006 / +0.031 / +0.081 (B-seq p<0.001). coverage_unc minus shuffled is negative in B.
- The DEV split-A gains (+0.11 to +0.14) did not replicate -> winner's curse on 10 seeds. Only plain coverage in B-seq passes Holm, and it is not attributable to the graph (shuffled control equally good); shuffled control for plain coverage not run.
- Next ideas (not run): shuffled/typeonly control for plain `typed_decisive_coverage`; per-type quotas; understanding why B (20% hold-out) differs from A; a model-side change (GCN reward model) needs an advisor decision.

## Phase 2 (after the failed confirmation): literature-driven plan, decided by Claude on user's delegation
Literature: low budget -> coverage/typicality beats uncertainty (TypiClust, Hacohen 2022; ProbCover, Yehuda 2022); labeler with blind spots -> weight by P(answerable) (Du & Ling, AAAI 2013 "knowledge blind spot"); abstention-aware active learning is a thin literature.
Our farthest-first coverage prefers outliers, which those papers argue against at low budget. New rule: `typed_decisive_probcover` (greedy weighted max cover in the propagated pair space, weight = P(decisive); radius = delta_q quantile of pairwise distances).
Protocol: DEV on seeds 400-409 (variants q02/q05/q10 + unc_q05, controls typeonly/shuffled of the chosen radius); freeze <= 3 candidates; CONFIRM on FRESH seeds 430-449 (never used), Random included in the run; Holm over candidates x 4 blocks; same success criterion.
Also pre-registered: the graph claim needs real minus shuffled AND real minus typeonly > 0; otherwise the gain is attributed to type-aware decisive weighting / coverage, not the graph.

### Phase 2 DEV result (seeds 400-409; log-loss gain; A-single / A-seq / B-single / B-seq)
probcover_q02 -0.009 / +0.059 / +0.015 / +0.065; q05 -0.037 / +0.012 / -0.004 / +0.097; q10 -0.002 / +0.041 / -0.002 / +0.071; unc_q05 +0.002 / +0.060 / +0.003 / +0.041.
Reference: typed_decisive_coverage (farthest-first) +0.095 / +0.109 / +0.024 / +0.105. ProbCover-style covering is NOT better than farthest-first on DEV -> dropped, no more tuning.

### FROZEN PHASE 2 TEST (written before seeds 430-449 were run; these seeds were never used)
Question: is the only confirmed signal of phase 1 (plain typed_decisive_coverage, B-sequential +0.083, B-single +0.062) real, and is it due to the graph?
- Candidate (single): typed_decisive_coverage. Holm over its 4 blocks (log-loss). Replication success: Holm p < 0.05 and positive in BOTH B-single and B-sequential, AUC gain >= 0.
- Graph claim (separate, requires replication first): paired difference coverage minus typed_decisive_coverage_shuffled > 0 and minus typed_decisive_coverage_typeonly > 0 (Wilcoxon p < 0.05 in B blocks). Otherwise attribute the gain to coverage / type-aware decisive weighting, not the graph.
- Non-candidate references run on the same seeds: core_set_relation, typiclust_pairs (embedding coverage baselines without graph) -> tells whether the graph rule beats plain coverage.
- Seeds 430-449, both splits, both conditions, Random in the same run.

### PHASE 2 RESULT (seeds 430-449, run once; `aggregate_phase2.py`; report §5.16): replication NOT met
typed_decisive_coverage log-loss gain A-single / A-seq / B-single / B-seq: +0.039 / +0.019 / +0.041 / +0.042 (Holm .26 / .43 / .29 / .43); B-split phase-1 effect (+0.062 / +0.083) did not replicate.
Controls: typeonly +0.033 / +0.046 / +0.042 / +0.030; shuffled +0.048 / +0.062 / +0.062 / +0.050 (shuffled is as good or better -> no graph effect). core_set_relation +0.072 / +0.072 / -0.027 / -0.048; typiclust_pairs ~ Random or worse.
Conclusion: no graph-attributable improvement. Common ingredient of real/typeonly/shuffled (type-aware decisive weighting + coverage) gives a consistent ~+0.03..+0.06, not pre-registered, not significant per block after Holm.
Untried: per-type quotas, joint multinomial outcome model / expected-information formulation, graph with unlabelled trajectory images, GCN as reward model (advisor decision), second encoder (GPU).

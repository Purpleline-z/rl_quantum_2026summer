# New method directions beyond graph selection: overnight run (branch `claude/new-directions-overnight`)

Run on 2026-10-10, 04:30-12:30 UTC, unattended, CPU only. Everything below is on the **same 168 labelled pair groups** as the earlier studies (re-split with new random seeds, not new data), with frozen SimCLR features and the judgment-unit protocol of `main` (one (pair, type) judgment per query, budgets 10 / 20 / 40 / 60 judgments, Splits A and B, the per-budget re-tuned head schedule, held-out image-disjoint test groups). Graph methods are not covered (another session does that). Log of every step: `OVERNIGHT_LOG.md`.

## 0. Summary

1. **One direction produced a clear, pre-registered, confirmed gain: weighting the reference-anchor loss.** The Bradley-Terry head is trained with a ranking loss on the labelled ideal (reference) images; the repository weights it 0.25. With weight 8 the held-out preference prediction improves on 35 fresh seeds in both splits (all 12 pre-registered tests Holm <= 0.003): AUC +0.012 (split A) / +0.019 (split B), accuracy +0.019 / +0.023, log-loss +0.052 / +0.065 versus the default. Using the class labels of all 144 ideal images of the split as anchors adds a little (AUC +0.020 / +0.024, accuracy +0.026 / +0.030). The same change raises the paper's own endpoint, type accuracy of the 28 outer-test ideal images, by +0.043 (pre-registration 2, Holm < 0.0001).
2. **The earlier selection result replicates with the default learner but is absorbed by the better learner.** With the default head `vopt_u` (the pool-wide I-optimal rule of the earlier study) beats random selection on fresh seeds (AUC +0.010 / +0.013, accuracy +0.016 / +0.022, Holm 0.004 / 0.006 / 0.0006 / 0.0006). With the anchor-weighted head the advantage of `vopt_u` over random is not confirmed (AUC +0.002 [-0.001, +0.006] in A, +0.005 [+0.001, +0.010] in B, Holm 0.75 / 0.20; accuracy +0.000 [-0.004, +0.005] / +0.002 [-0.003, +0.008], Holm 1.0). A cheap change of the learner is worth more than the selection rule, and the earlier comparisons of acquisition rules were made against a weakly tuned learner.
3. **The other nine ideas tested (the five I proposed and four new ones) did not beat the baseline on the development seeds**; none was promoted to confirmation (details in sections 3 and 4). Several of the negative results are informative: a time-contrastive adapter on the trajectory frames hurts; filename metadata (temperature, frame index, time of day) does not help; tie and not_apply judgments change the score scale (raw log-loss) but not ranking or temperature-calibrated log-loss; coupling the four type heads adds nothing; a GP preference learner is better in split A (log-loss) and equal in split B.
4. **Honest limits.** Same 168 groups, so the Wilcoxon p-values are optimistic; the anchor result is for a frozen-feature head - whether it carries over to the end-to-end fine-tuned model of Task 3c needs a GPU run (a recipe is in section 6); the weight 8 was chosen on development seeds (winner's curse), the confirmatory run is the evidence; the literature was checked only through search results (no page could be opened).

## 1. Protocol and what counts as evidence

- **Seeds.** Exploration used development seeds 2000-2019 only. Confirmatory seeds 3000-3034 are disjoint from every seed of the earlier work and from the development seeds (asserted in `nd_core.py` and in `test_new_directions.py`).
- **Paired design.** Learners are compared on identical random label sets (3 draws per seed and budget); selectors on the identical initial model and pool. Gains are means over budgets and draws, then over seeds; intervals are paired bootstrap over seeds; log-loss gain = reference - cell, so positive is better.
- **Pre-registration.** `PREREGISTRATION_1.md` + `prereg1_spec.json` (commit `b1669ca9`) and `PREREGISTRATION_2.md` (commit `5a2dba20`) were committed before their confirmatory cells existed. The tests are two-sided Wilcoxon signed-rank over 35 seeds with Holm correction within the families given there (`nd_confirm.py`, `nd_confirm_type.py`). Everything not in a pre-registration is labelled development or exploratory.
- **Integrity.** Eight tests (`code_behavior_tests/test_new_directions.py`): seed sets, held-out images never in labels, predictions independent of test labels, mirror head exactly invariant, selectors ignore hidden outcomes, selections distinct, stochastic batch reproducible, GP/Bayes predictors finite and antisymmetric. The ideal images used as extra anchors share no content identity with any pair image (checked by SHA-256 for seed 2000 in both splits: 144 ideal vs 262 / 269 pair-image identities, overlap 0).
- **Tuning of the comparator.** The default anchor weight 0.25 is part of the baseline learner; the result of section 2 says the baseline was not tuned for this endpoint. Following Lüth et al. (NeurIPS 2023), where apparent gains of acquisition rules often come from poorly tuned baselines, every new comparison below should be made against the tuned learner.

## 2. Confirmed: anchor-weighted multi-task Bradley-Terry learning

**Method.** The head loss is the repository's: Bradley-Terry for decisive judgments, |a-b| for ties, a push-down term for not_apply, a ranking loss among the labelled ideal images of the four types (reference anchors) with weight `aw`, and a bad-image term. Only `aw` changes (0.25 -> 8). Variant `aw8_all` also uses the class labels of the seed's utility-validation and outer-test ideal images as anchors (they are absolute labels the lab owns; the preference endpoint never evaluates ideal-image classification, and ideal images are excluded from the pair universe by content identity). `aw8_all` cannot be evaluated on the type-accuracy endpoint, which is why that endpoint uses `aw8` only.

**Confirmatory results (pre-registration 1, seeds 3000-3034; `results/CONFIRM1_TABLE.md`, generated).**

| family | split | metric | cell A - cell B | seeds | gain [95% interval] | seeds better | p | Holm |
|---|---|---|---|---|---|---|---|---|
| F1 | A | auc | random:aw8 - random:baseline | 35 | +0.0122 [+0.0054, +0.0192] | 77% | 0.0015 | 0.0030 |
| F1 | A | acc | random:aw8 - random:baseline | 35 | +0.0186 [+0.0085, +0.0288] | 69% | 0.0020 | 0.0030 |
| F1 | A | ll | random:aw8 - random:baseline | 35 | +0.0524 [+0.0349, +0.0707] | 89% | < 0.0001 | < 0.0001 |
| F1 | B | auc | random:aw8 - random:baseline | 35 | +0.0194 [+0.0119, +0.0281] | 89% | < 0.0001 | < 0.0001 |
| F1 | B | acc | random:aw8 - random:baseline | 35 | +0.0230 [+0.0129, +0.0342] | 80% | < 0.0001 | 0.0002 |
| F1 | B | ll | random:aw8 - random:baseline | 35 | +0.0654 [+0.0449, +0.0886] | 89% | < 0.0001 | < 0.0001 |
| F1 | A | auc | random:aw8_all - random:baseline | 35 | +0.0197 [+0.0126, +0.0274] | 86% | < 0.0001 | < 0.0001 |
| F1 | A | acc | random:aw8_all - random:baseline | 35 | +0.0264 [+0.0165, +0.0367] | 74% | < 0.0001 | < 0.0001 |
| F1 | A | ll | random:aw8_all - random:baseline | 35 | +0.0752 [+0.0548, +0.0966] | 86% | < 0.0001 | < 0.0001 |
| F1 | B | auc | random:aw8_all - random:baseline | 35 | +0.0235 [+0.0134, +0.0356] | 80% | < 0.0001 | < 0.0001 |
| F1 | B | acc | random:aw8_all - random:baseline | 35 | +0.0298 [+0.0190, +0.0422] | 86% | < 0.0001 | < 0.0001 |
| F1 | B | ll | random:aw8_all - random:baseline | 35 | +0.0748 [+0.0494, +0.1046] | 91% | < 0.0001 | < 0.0001 |
| F2 | A | auc | vopt_u:aw8 - random:aw8 | 35 | +0.0022 [-0.0013, +0.0057] | 69% | 0.2516 | 0.7549 |
| F2 | A | acc | vopt_u:aw8 - random:aw8 | 35 | +0.0002 [-0.0042, +0.0045] | 51% | 0.8980 | 1.0000 |
| F2 | B | auc | vopt_u:aw8 - random:aw8 | 35 | +0.0051 [+0.0008, +0.0097] | 60% | 0.0495 | 0.1978 |
| F2 | B | acc | vopt_u:aw8 - random:aw8 | 35 | +0.0024 [-0.0031, +0.0082] | 49% | 0.5382 | 1.0000 |
| F3 | A | auc | vopt_u:baseline - random:baseline | 35 | +0.0100 [+0.0045, +0.0157] | 69% | 0.0022 | 0.0044 |
| F3 | A | acc | vopt_u:baseline - random:baseline | 35 | +0.0157 [+0.0091, +0.0222] | 77% | 0.0001 | 0.0006 |
| F3 | B | auc | vopt_u:baseline - random:baseline | 35 | +0.0134 [+0.0057, +0.0219] | 63% | 0.0060 | 0.0060 |
| F3 | B | acc | vopt_u:baseline - random:baseline | 35 | +0.0221 [+0.0121, +0.0328] | 77% | 0.0002 | 0.0006 |
| F4 | A | auc | vopt_u:aw8 - random:baseline | 35 | +0.0144 [+0.0072, +0.0220] | 80% | 0.0003 | 0.0006 |
| F4 | A | acc | vopt_u:aw8 - random:baseline | 35 | +0.0188 [+0.0084, +0.0295] | 66% | 0.0016 | 0.0016 |
| F4 | A | ll | vopt_u:aw8 - random:baseline | 35 | +0.0564 [+0.0377, +0.0757] | 91% | < 0.0001 | < 0.0001 |
| F4 | B | auc | vopt_u:aw8 - random:baseline | 35 | +0.0245 [+0.0148, +0.0352] | 71% | < 0.0001 | 0.0002 |
| F4 | B | acc | vopt_u:aw8 - random:baseline | 35 | +0.0254 [+0.0140, +0.0374] | 74% | 0.0002 | 0.0006 |
| F4 | B | ll | vopt_u:aw8 - random:baseline | 35 | +0.0694 [+0.0427, +0.0986] | 80% | < 0.0001 | < 0.0001 |

(The 35 seeds are the units; each gain is the mean over budgets 10-60 and 3 random draws; 69-91% of seeds are better in every F1 test. F1: aw8 or aw8_all vs the default head with random labels; F2: does `vopt_u` still beat random with the aw8 head; F3: `vopt_u` vs random with the default head; F4: `vopt_u` + aw8 vs random + default head. The table is generated by `nd_confirm_markdown.py` from the result files.)


**Exploratory breakdown of the confirmatory cells (not pre-registered; `results/confirm1_per_type_exploratory.csv`).** By budget the AUC gain over the default head shrinks from the smallest budget to the largest: split A, `aw8` +0.019 (10 judgments), +0.014, +0.009, +0.007 (60); split B +0.032, +0.022, +0.008, +0.016. By type (accuracy on the type's own held-out judgments, mean gain over the default, 95% bootstrap interval over seeds): (1x1) +0.025 [+0.003, +0.047] (A), +0.025 [+0.006, +0.045] (B); sqrt13 +0.024 [+0.014, +0.035] / +0.026 [+0.014, +0.040]; c(6x2) +0.001 [-0.017, +0.019] / +0.025 [+0.001, +0.050]; **HTR +0.010 [-0.021, +0.041] / +0.004 [-0.016, +0.023] for `aw8`, but +0.036 [+0.014, +0.059] / +0.026 [+0.005, +0.048] for `aw8_all`**. The lab's priority type (HTR) therefore gains only when all ideal images are used as anchors (HTR has the fewest reference images: 16 of the 88 training references, 26 ideal HTR images in total after de-duplication).

**Type accuracy of the outer-test ideal images (pre-registration 2, `results/CONFIRM2_TABLE.txt`).** Mean over budgets 0-60 and 3 draws, default 0.25 vs weight 2 / 8 / 32: +0.030 [+0.019, +0.041], +0.043 [+0.030, +0.057], +0.047 [+0.033, +0.064] (Holm <= 0.0001; 77-83% of seeds better). Levels rise from 0.78-0.81 to 0.83-0.85. This is still below the frozen 1-NN to the references (0.871 on the five paper seeds) and below the simple controls of §5.19 of the report, so the anchor weight closes part of the gap that the frozen head had; it does not make the Bradley-Terry head better than the controls on this endpoint.

**Development evidence behind the choice (20 dev seeds; `results/DEV_SCREENS.md`).** Weight 0: AUC -0.023, accuracy -0.012 to -0.015, log-loss -0.12 to -0.20 (anchors matter). Weights 1-16 improve over 0.25 (AUC gains; weight 1 from an 11-seed screen; split A +0.007 at 1, +0.016 at 2, +0.021 at 4, +0.024 at 8, +0.024 at 16; split B +0.004, +0.007, +0.007, +0.008, +0.004); weight 32 loses the gain (A +0.016, B -0.006). The largest gain is at the smallest budgets (weight 8, AUC gain at 10 / 20 / 40 / 60 judgments: split A +0.045 / +0.028 / +0.014 / +0.011, split B +0.015 / +0.014 / +0.001 / +0.002) and shrinks with more labels. The bad-image weight (0 / 0.1 / 1) and a 1/N-scaled anchor weight change nothing; more steps or a higher learning rate add about +0.003 to +0.01 accuracy but cost log-loss; a lower learning rate hurts. A feature-space "mixture-consistency" anchor (synthetic mixtures of two ideal images, ordered by mixing share) added nothing on top of weight 8 (section 4).

**Why it plausibly works (hypothesis, not tested).** Pair judgments say which image is better for a type; the 144 ideal images say what each type looks like. With 10-60 pair judgments the second signal is far more plentiful, and the default weight lets the pair loss dominate the 4 x 256 head weights early in training. This is the same multi-task weighting question as in Kendall et al. 2018 and Shi et al. 2020 (auxiliary-task reweighting for minimum-data learning); we found no paper that sweeps this weight for pairwise preference models, so these are motivation, not support.

**What it changes for earlier conclusions.** (i) Acquisition rules compared with a weakly weighted learner may have been credited with gains that a better learner provides: with the tuned head the selection advantage of `vopt_u` is within noise. (ii) The earlier statement that type accuracy is "saturated by the anchors" is partly an artefact of their low weight. (iii) The earlier conclusion that no uncertainty rule beats random is not affected.

**Open question that needs a GPU.** In the end-to-end fine-tuned model the anchor term is one random reference pair per mini-batch with weight 0.25 (`Experiment.train`); the corresponding change is to raise that constant. I did not edit the pipeline: the constant is a literal in `Experiment.train`, and adding a `Config` field could change the hashes used to resume Task 3c cells. Recipe in section 6.

## 3. The five directions proposed earlier

| # | Direction | What was implemented | Development result (20 seeds unless stated; gains vs the default head with the same random labels) | Verdict |
|---|---|---|---|---|
| 1 | Type-coupled preference model (+ joint design) | Prior-centred last layer on the anchor-trained hidden layer; per-type deviations plus a shared deviation across types; penalties chosen by grouped CV (`nd_bayes.py`) | Coupled vs independent: no difference (split A AUC +0.001 vs -0.003; B -0.011 vs -0.015, 10 seeds); both at or below the MLP head. | Not supported. Coupling is never chosen over independent types (CV), consistent with not_apply being type-specific. |
| 2 | Bayesian / schedule-free head | Same head; CV-chosen penalty (grouped 5-fold on the labelled judgments); separate CV choice of (lr, steps) for the MLP head | Bayes head: AUC -0.003 / -0.015 (A / B); CV-chosen schedule (4 seeds): -0.007 / +0.002 | Not supported. CV on 30-100 judgments is too noisy to beat the fixed re-tuned schedule. |
| 3 | Goal-oriented (transductive) design | `vopt_u` with target weights = type-ambiguity of the judgment's images (entropy of the win-rate type posterior of the anchor head), ambiguity-weighted test metric | vs random selection: AUC +0.012 (A), +0.009 (B), same as `vopt_u` (+0.013 / +0.009); on the ambiguous half of the held-out judgments no gain either | Not supported. The ambiguity target adds nothing to the plain pool target. Real downstream evaluation (type labels of trajectory frames) is impossible: the absolute-label file has 18 "Bad" labels and no type labels. |
| 4 | Time-contrastive adapter on trajectory frames | Linear adapter (512 -> 128) with InfoNCE on frames of the same run within +-3 indices, trained on all 1124 trajectory frames except the held-out images of the seed; head sees [x, g(x)] or g(x) | AUC -0.008 (A) / -0.012 (B); log-loss -0.10; adapter alone -0.000 / -0.012 | Negative: hurts. Plausible reasons (not tested): neighbouring frames are near-duplicates, and temporal neighbours straddle type boundaries. |
| 5 | Ties and not_apply | Drop non-decisive rows; drop only ties; drop only not_apply; Rao-Kupper likelihood with per-type threshold; temperature-calibrated check on split A | Dropping non-decisive rows: AUC -0.002 / -0.005 (n.s.), accuracy +0.000 / +0.004, but raw log-loss -0.19 / -0.16; after fitting one temperature on validation groups the log-loss difference is -0.056 [-0.167, +0.008] (n.s.). Rao-Kupper: AUC -0.003, log-loss -0.06 (11 seeds). | Informative negative: ties / not_apply rows set the score scale, not the ranking. Rao-Kupper does not help. |

## 4. Five new method directions (plus extras)

Support = what the (search-verified, never page-verified) literature says; table with citations in `literature/LITERATURE_TABLE.md` and the full lists in `literature/lit_*.md`. All results are development seeds (paired, random labels) unless stated.

| # | Direction | Literature support | Implementation | Result | Beats the baseline? |
|---|---|---|---|---|---|
| N1 | **Anchor-weighted multi-task BT learning (+ all absolute labels as anchors)** | Indirect: uncertainty weighting (Kendall et al. CVPR 2018), GradNorm (ICML 2018), auxiliary-task reweighting for minimum data (Shi et al. NeurIPS 2020); absolute + comparison labels (Guo et al. IJCAI 2018, Compare2Score NeurIPS 2024); caution Kurin et al. NeurIPS 2022 | `aw8`, `aw8_all` (`nd_learners.py`) | Section 2: confirmed on 35 fresh seeds | **Yes** (learner level; also the default-head selection gain disappears) |
| N2 | GP preference learner on frozen features | Chu & Ghahramani ICML 2005; Houlsby et al. 2011 (BALD for preference learning); BAL-PM NeurIPS 2024 | `nd_gp.py`: one RBF-kernel latent per type over all 1278 images, MAP in the dual, penalty by grouped CV, anchors as in the head | 10 seeds: split A log-loss +0.060 [+0.001, +0.119], AUC +0.013 [-0.003, +0.028]; split B ~0 (AUC +0.002). GP-variance acquisition: AUC +0.004 / -0.007 (n.s.) | Partly (A only, not significant on AUC); not promoted |
| N3 | Mirror-symmetry prior (head and acquisition) | Consistency-based AL: Gao et al. ECCV 2020, CAMPAL ICML 2023, LADA NeurIPS 2021; symmetry as a prior: NequIP (indirect only) | Head averaged over the image and its mirror (exactly invariant, tested); acquisition = prediction disagreement between the image and its mirror, alone or multiplied into `vopt_u` | Head: accuracy +0.009 [+0.002, +0.015] (A), +0.004 (B, n.s.), AUC -0.000 / -0.005; consistency selection: AUC +0.000 / -0.007; `vopt_cons` ~ `vopt_u` | No |
| N4 | Mixture-consistency anchors (physics: real patterns ~ linear mixtures of ideal patterns) | mixup (Zhang et al. ICLR 2018), Manifold Mixup (ICML 2019), RankMixup (ICCV 2023, supervises order only); linear-mixture models for XRD (NMF; Lee et al. Nat. Commun. 2020 synthetic mixtures); no RHEED precedent found | `fit_head_ext`: ordered pairs of feature-space mixtures of two ideal images (share 0.8 vs 0.3) must be ordered for the types involved | On top of weight 8 (all ideal anchors, 16 seeds): AUC +0.031 vs +0.031 (A), +0.011 vs +0.014 (B) at weight 1; worse at weight 4; with default anchor weight + mixture weight 4: +0.009 (A), -0.012 (B) | No |
| N5 | Pseudo-label self-training of preferences | SSRM (Findings of EMNLP 2024), FixMatch NeurIPS 2020, confirmation bias (Arazo et al. IJCNN 2020), NeST AAAI 2023 | Label confident unlabelled pool judgments with the current head (threshold 0.9, weight 0.5), retrain | AUC -0.001 / -0.001; accuracy +0.003 / +0.005 (B, CI > 0); log-loss -0.04 | No |
| N6 | Fusion of filename process metadata (the lab's open item) | DAFT (MICCAI 2021), FiLM (AAAI 2018), HyperFusion (MedIA 2025); no RHEED + temperature fusion found | Concatenate [temperature, frame index / frames in the run, time of day, present-flag] to the SimCLR vector (zeros for ideal images), same head | AUC -0.004 [-0.007, -0.001] (A), -0.004 (B); log-loss -0.02 | No (slightly negative). Only 4 of the lab's wished-for variables exist in file names; pressure and exposure are not there. |

Extras (development, not promoted):
- **Design variants of `vopt_u`** (stochastic batches after Kirsch et al., TMLR 2023; design in raw feature space; ambiguity or mirror-consistency factors): all within noise of `vopt_u` (gain over random with the default head, AUC: split A +0.008 to +0.016 against +0.013 for `vopt_u`; split B -0.003 to +0.007 against +0.007 to +0.009). No variant is reliably better.
- **Ensembles of 5 or 10 heads:** AUC +0.001 (10 seeds): head variance is not the bottleneck.
- **Selection with the anchor-weighted model** (`vopt_aw8_all`, look at the model trained with weight 8): not better than `vopt_u` (split A AUC +0.033 vs +0.034 with the same final learner; B +0.025 vs +0.021).
- **Committee of MLP head and GP as acquisition:** implemented (`qbc_gp`, `vopt_qbc`) but not completed (CPU time).
- **Cold start, anchors as cross-entropy, combinations of the anchor-weighted head with the other ideas:** see section 5 (pending at the time of writing this paragraph; filled in below if finished).

## 5. Later additions (cold start, cross-entropy anchors, combinations)

PENDING

## 6. How to use the finding, and what needs a GPU

- **Frozen-head studies (CPU):** set the anchor weight to 8 (or use `aw8_all` when ideal-image classification is not the endpoint). `nd_learners.py::aw8`.
- **End-to-end fine-tuning (GPU, untested):** in `Experiment.train` the anchor term is `.25 * -logsigmoid(...)` for one random reference pair per mini-batch. Replace `.25` by a larger constant (start with 8, also try 2 and 32), keep everything else, and compare with the 0.25 run on the five paper seeds and on the held-out preference endpoint (`pair_preference_endpoint.evaluate_full`). A one-line `Config` field would change the cell hashes of existing Task 3c results, so use a separate output folder.
- **Reproduce:**
  `cd code/active_learning_studies/new_directions; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1`
  development learner screens `python3 nd_run.py --learners baseline,aw8,aw8_all --seeds 2000-2019 --splits A,B --draws 3 --out results/screenX`;
  selection screens `python3 nd_select.py --selectors random,vopt_u --learners baseline,aw8,aw8_all --seeds 3000-3034 --splits A,B --draws 3 --out results/confirm1`;
  confirmatory tests `python3 nd_confirm.py prereg1_spec.json results/confirm1`; type accuracy `python3 nd_type_accuracy.py 3000-3017 out.json` and `nd_confirm_type.py`; tables `python3 nd_make_report_tables.py`. The SimCLR feature cache (`results/frozen_encoder_task3/simclr_feature_cache.pt`) and `new_directions/results/feature_cache_all.pt` are rebuilt with `build_feature_cache.py` and `nd_temporal.py build` (both are small; the first is not committed).

## 7. Caveats to read before using any number

1. Seeds re-split the same 168 pair groups and 144 ideal images; the held-out sets are small (about 50-60 decisive judgments in split A, about 100 in split B). Wilcoxon p-values are optimistic; the intervals are over seeds.
2. Everything is a frozen SimCLR head; the encoder was never fine-tuned here.
3. The anchor weight, the schedule and the grid were explored on the development seeds; the confirmatory run tests the chosen setting. Other combinations in section 5 are development only.
4. Literature: titles, authors and venues were confirmed only from search results; claims about the content of a paper come from abstracts and summaries. Items marked UNVERIFIED / PARTIAL in `literature/` were not relied on.
5. Several negative results (time-contrastive, GP, Bayes, pseudo-labels) could be sensitive to implementation choices I did not tune (adapter size and window, kernel family, thresholds); they say "no gain in this implementation", not "impossible".

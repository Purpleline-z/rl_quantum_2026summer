# Handoff for the next Claude: graph-aware active selection (branch `claude/graph-active-selection`)

Written 2026-10-09 (UTC afternoon) at the end of an autonomous run that was interrupted repeatedly by container restarts. Read this file first, then `AUTONOMOUS_RUN_LOG.md` (round-by-round log, the pre-registered protocol) and `GRAPH_PLAN.md`.

## 0. Who and how
- User: purpleline@uchicago.edu, PhD student; writes Chinese; reply in Chinese, plain language, no unexplained jargon, hedged and honest (no overclaiming; separate findings from hypotheses). Advisor-level contact "Justin Meng" (PhD student advisor-like role) wants the graph idea pushed: "simple graph-aware acquisition vs SimCLR-embedding methods first, then GCN; novelty from pairwise preference + type-aware acquisition + trajectory structure; may help RLHF".
- Repo `Purpleline-z/rl_quantum_2026summer`. Branches: `main` untouched; `claude/judgment-unit-rerun` (judgment-unit study, parent of this branch); `claude/frozen-encoder-strategies` (earlier work; its `SESSION_CONTEXT_FOR_NEXT_CLAUDE.md` is the background document for the whole project, also present on this branch at the repository root); **this branch `claude/graph-active-selection` holds all graph work**. No PR, nothing merged.
- The user said: large GPU tasks are theirs (list: `GPU_TASKS_FOR_USER.md`); small CPU experiments and literature reading are yours. The fixed protocol must not change (see section 2).

## 1. Project in one paragraph
RHEED reconstruction images (types (1x1), c(6x2), (sqrt13 x sqrt13)=RT13, HTR; Twinned excluded). A Bradley-Terry reward head (frozen SimCLR ResNet-18 features, one output per type) is trained from expert pairwise judgments; the question is which judgments to label next. 168 usable pair groups, 521 judgments (51% decisive, 14% tie, 35% not_apply), 284 images. Endpoint: held-out preference prediction on decisive judgments of held-out pair groups (log-loss, AUC).

## 2. Fixed protocol (do not change)
Judgment-unit study (`judgment_unit_study.py`): query unit = one (pair, type) judgment; budgets 10/20/40/60 judgments; splits A (10 initial / 20 validation / 40 test groups) and B (classifier2-style 20% hold-out); single-shot and sequential (rounds of 10); head schedule `results/pair_endpoint_study/schedule.json`; Random = mean of 5 draws; gain = per-seed difference to Random averaged over budgets; Wilcoxon over seeds + Holm; seeds 42, 79, 123, 202, 303, 400-429 (35). Only selectors may change. A selector sees candidate judgments, revealed judgments, the model, cached features, and (this run) the typed reference images (`exp.references`, already training anchors of every strategy); never test/utility images or validation/test group images.
Selection-bias rule decided before the new methods ran: tune on DEV seeds 400-409 only; freeze at most 4 candidates; run CONFIRM seeds 410-429 + 42, 79, 123, 202, 303 once. Success = a frozen candidate with Holm p < 0.05 (log-loss gain over Random) in at least two of the four blocks incl. both splits, non-negative AUC gain. Not yet evaluated.

## 3. What has been established (details: report section 5.14 in `code/ACADEMIC_REPORT_DRAFT.md`, `AUTONOMOUS_RUN_LOG.md`)
1. The comparison graph is nearly a matching (168 edges, 284 nodes, 117 components; per type, components of <= 4-5 nodes), so a GNNRank-style ranking GNN on comparison edges has nothing to recover. The slide's "669 pairs / ~300 images" does not match the data.
2. First-generation graph rules (`graph_strategies.py`: PageRank/boundary-weighted uncertainty, SGC core-set, shuffled controls), 35 seeds: nothing significant after Holm; weak split-A hint for PageRank-weighted uncertainty, not replicated in B (report 5.14).
3. Data-level diagnostic (all 521 rows; design-informing): label propagation of the typed reference images over a kNN graph (k=10, alpha .9) gives 8 compact features per judgment; AUC for decisive 0.83 (type only 0.60, type+cosine 0.71, 1024-d mean+|a-b| 0.80), tie 0.88, not_apply 0.90. Only k in {5, 10} and alpha .9 were tried (k=5 worse).
4. Type-aware rules (`graph_typed.py`) on DEV seeds 400-409, log-loss gain over Random in blocks (A-single, A-seq, B-single, B-seq), 10 seeds only (noisy; shares of seeds better 60-100%):
   - typed_decisive_coverage_unc (farthest-first coverage in SGC-propagated pair space x P(decisive) x (0.25 + own-head uncertainty)): +0.108 / +0.138 / +0.029 / +0.091; AUC +0.019 / +0.022 / +0.005 / +0.020. Best so far.
   - typed_decisive_coverage: +0.095 / +0.109 / +0.024 / +0.105; ..._g2 (P^2): +0.132 / +0.120 / +0.005 / +0.065.
   - control `typeonly` (decisive predictor from type one-hot only): +0.082 / +0.101 / +0.022 / +0.056, AUC ~ 0. So **most of the log-loss gain is type-aware decisive weighting, not the graph**; the graph features add about +0.01 AUC and little log-loss (largest in B-seq, +0.05). Be explicit about this in any write-up.
   - Literature graph baselines (`coregcn`, `uncertaingcn`, implemented from the authors' code logic) are not better than the embedding baselines (core_set_relation, typiclust_pairs, bald_decisive) on DEV.
   - Decisive-weighted random sampling is bad (B-single -0.06 to -0.10): coverage is needed. Split B single-shot is hard for every rule.
5. Round 3 (variants of coverage_unc: gamma 2, uncertainty power 2, q-coordinates, plus its `typeonly` and `shuffled` controls) was running on DEV seeds when this was written; results are in the cell files (see section 5 to aggregate).

## 4. Where things are (all under `code/active_learning_studies/pair_disjoint_not_image_disjoint/`)
- `graph_typed.py` (type-aware rules, `ImageGraph`, `decisive_probability`, GCN baselines; `TYPED_NAMES`), `graph_strategies.py` (first-generation graph rules), registered lazily in `judgment_unit_strategies.make_selector`; `JU_GRAPH=1` adds them to `judgment_unit_study.strategy_names()`; `JU_ONLY=a,b,c` restricts the run.
- `aggregate_dev.py LO-HI [names]` (dev table, no Holm), `aggregate_graph_vs_random.py` (35-seed Holm tables for the first-generation rules), `make_graph_report_section.py` (generates report 5.14; `--insert`), results in `results/judgment_unit_study/{A,B}_groups/{single,sequential}/seed<N>_<strategy>.json`.
- Tests: `code/active_learning_program/code_behavior_tests/test_graph_strategies.py` (+ the older ones; run `python3 -m pytest active_learning_program/code_behavior_tests -q` from `code/`; needs `pip install pytest matplotlib`; **running the old tests rewrites `results/active_learning_v1.8_seed42/manifests`: restore with `git checkout -- <that folder>` before committing**).
- `graph_exploration/`: this file, `AUTONOMOUS_RUN_LOG.md`, `GRAPH_PLAN.md`, `notes_graph_literature_round.md`, `GPU_TASKS_FOR_USER.md`, diagnostic scripts (`graph_structure.py`, `graph_signal_test.py`, `boundary_structure.py`), `scripts/launch_round.sh`, `scripts/autocommit.sh`.

## 5. How to run (CPU, 4 cores; 2 threads per process)
```
cd code/active_learning_studies/pair_disjoint_not_image_disjoint
graph_exploration/scripts/launch_round.sh r4 typed_decisive_coverage_unc,typed_decisive_coverage_unc_typeonly 400-409   # both splits, finished cells skipped
graph_exploration/scripts/autocommit.sh &                                                                             # commits result cells while jobs run
python3 aggregate_dev.py 400-409 typed_decisive_coverage_unc,typed_decisive_coverage_unc_typeonly,typed_decisive_coverage_unc_shuffled,...
```
About 2-3 minutes per seed per split for ~5-8 strategies (both conditions). Needs `pip install torch scikit-learn scipy matplotlib pytest` if the container is fresh; the feature cache `results/frozen_encoder_task3/simclr_feature_cache.pt` is NOT in git (rebuild: `PYTHONPATH=../../active_learning_program python3 build_feature_cache.py`, ~minutes).

## 6. Pitfalls learned
- **Container restarts kill background jobs** (happened ~4 times; uptime shows minutes). Check `uptime` and `ps -eo pid,etime,cmd | grep [j]udgment_unit_study`; relaunch with `launch_round.sh` (idempotent). Files persist; processes do not. Schedulers (`send_later`) were disabled for the organisation; a session CronCreate check-in only works while the session is open.
- The repo's stop hook demands a clean tree: commit result cells often (autocommit script), but `git add` only specific paths (never `git add -A code`: it picks up rewritten manifests).
- arxiv.org, ar5iv and openaccess.thecvf.com are blocked by the network proxy; GitHub raw works via WebFetch; WebSearch works.
- Never `pkill -f pattern` from the shell. Use 2 torch threads. Do not tune on CONFIRM seeds.
- `pytest` collection fails without matplotlib.

## 7. Next steps (in order)
1. Finish DEV round 3 (relaunch if dead), aggregate with `aggregate_dev.py 400-409`, log in `AUTONOMOUS_RUN_LOG.md`.
2. Freeze at most 4 candidates in the log (suggest: the best coverage_unc variant, its `typeonly` control and its `shuffled` control are controls, not candidates; add typed_decisive_bald as a second candidate), then run CONFIRM seeds once: `launch_round.sh conf <names>,random,uncertainty 410-429` and a second call for `42,79,123,202,303` (Random cells exist for all 35 seeds). Aggregate with Holm (extend `aggregate_graph_vs_random.py` to list the new names; it currently has only the first-generation names in `CONTROLS`).
3. Write report section 5.15 (generate tables from cells; state that most gain is type-aware decisive weighting, the graph part is the AUC/log-loss increment over `typeonly`) and update the abstract/conclusion only after the confirmation result is known. Report failure honestly if the success criterion is not met.

## 8. Directions NOT yet explored (candidates for the next session; all CPU unless noted)
- **Type-aware budget allocation**: the `typeonly` control gets most of the gain, so how the budget is split over types (and over decisive-likely pairs) may be the real lever; try explicit per-type quotas proportional to (type frequency in the held-out set) x (decisive rate) and compare with coverage_unc.
- **Better decisive/tie/not_apply model**: only a 9-feature ridge logistic was used; try a joint multinomial (decisive/tie/not_apply), calibrated models, or a hierarchical per-type model; use an expected-information formulation (value of a decisive vs tie vs not_apply outcome for the head's loss) instead of P(decisive) x uncertainty. Related to abstention-aware / noisy-oracle active learning (not searched).
- **Graph construction**: k and alpha were only tried on {5,10} / .9; multi-scale posteriors, mutual-kNN, local-scaling kernels, including the 840 unlabelled trajectory images and Yao's ~100 new trajectories as unlabelled nodes (needs the label-free / identity-safety discussion: held-out images must not be reachable from labelled set or pool), temporal chain edges within a session (needs physics-team confirmation; `data/temporal_constraints.json` marks them tentative).
- **Selection in the head's own hidden space** (graph-propagated hidden features, Fisher/D-optimal with graph-smoothed features) and batch-aware BALD with graph kernels; GRAIN/FeatProp-style influence or K-medoids on propagated features (read only via search summaries); AGE/ANRMAB-style adaptive combination of criteria.
- **Trained GCN as selector or as model**: CoreGCN/UncertainGCN here use a labelled-vs-pool discriminator on SimCLR features with default hyper-parameters (lr 1e-3, 200 steps, lambda 1) and were not tuned; a semi-supervised GCN reward model changes the evaluated model (protocol decision with the advisor; GPU for end-to-end).
- **Second encoder** (ImageNet / DINOv2 / SimCLR re-pretrained with the new trajectories): GPU, in `GPU_TASKS_FOR_USER.md`; a cached ImageNet ResNet-18 and SimCLR feature set for all 1278 images already exists in `image_representation_analysis/results/representation_exploration/cache/`.
- **RLHF-facing literature** (not yet searched): multi-attribute / multi-objective reward models with type-conditioned preference queries, active reward modelling with abstention, D-optimal design on graph features (Active Reward Modeling, ICML 2025, is in the older notes).
- **Statistics**: the DEV tables use raw p; with 25 confirmation seeds a gain below about 0.04 log-loss is unlikely to be significant (per-seed sd 0.09-0.14); consider reporting paired bootstrap intervals alongside Holm.
- **Open decision for the user**: whether selectors may see the typed reference (ideal) images. This run assumed yes (they are already training anchors); say so in any report.

# Pre-registration R1: does a different frozen representation lift `lap3_k10_st4`? (written before any confirmation seed 8650–8684 was run)

Branch `claude/representation-20261013`. Development plan: `PREREGISTRATION_R0_development_plan.md`; development seeds 8600–8619 (stage A: 8600–8609, 12 learners; ensemble control on 8600–8609 draw 0; stage B1 graph-free learners on all representations; stage B2 learning-rate re-tuning, stopped early with 4 complete seeds per split). Development numbers are optimistic and are not evidence.

## 1. Protocol extension (stated as required)
The learner may use: other layers of the frozen SimCLR ResNet-18 encoder (layer 2/3 pooled), the cached ImageNet-ResNet-18 features of `image_representation_analysis` (1278 images; no weights, **HuggingFace, torch hub and download.pytorch.org answer HTTP 403 through the proxy, so no other pretrained encoder can be used**), deterministic views (horizontal mirror, centre crop and four corner crops of 90 % side) averaged at feature level, and centring / whitening / PCA with statistics of the allowed graph nodes. No fine-tuning; no labels or images of validation / test pairs groups or the seed's outer-test ideal images in nodes, statistics or labels (SHA-256 exclusion as `nd_graph_learner.GraphData`; unit tests). Everything else (loss, anchors with weight 8, per-budget schedule with 4x steps, kNN graph k = 10, Laplacian lambda = 3) is `lap3_k10_st4`; the representation `s4` reproduces it to the last digit (test). Blocks are put on the scale of the baseline feature (see `nd_representation.py`); the learning rate and steps were **not** re-tuned per representation in the confirmation (a development check of lr x0.5 / x2 on 4 seeds per split showed changes within +-0.005 AUC for the baseline and for three representation learners).

## 2. Frozen candidates (4), all with head and graph from the stated representation unless noted
| id | learner | what changes |
|---|---|---|
| C1 | `rp_gs4_his` | graph from SimCLR final feature (as baseline); head input = ImageNet ⊕ SimCLR (per-image L2-normalised blocks, norm-matched, 1024-d) |
| C2 | `rp_i4` | ImageNet-ResNet-18 feature alone (512-d, norm-matched) for head and graph |
| C3 | `rp_s3` | SimCLR layer-3 pooled feature (256-d, norm-matched) for head and graph |
| C4 | `rp_mc` | mean of the SimCLR final feature over 7 deterministic views (original, mirror, centre crop, four corner crops) for head and graph |
Dropped after development (clearly worse, not frozen): PCA-whitened 64/128 (-0.04…-0.07 AUC), ImageNet ⊕ SimCLR for head and graph, layer3 ⊕ final, mirror average, graph from ImageNet ⊕ SimCLR with SimCLR head.
**Reference learner**: `lap3_k10_st4`. **Tuned graph-free control**: `aw8_all_st2lr0.5`. **Controls of each candidate** (shuffled graph `rp_..._shuf`; capacity control `rp_cap` = SimCLR final ⊕ fixed random 512-d projection of it): they are run **only for candidates whose AUC and accuracy differences to the reference are positive (point estimates) in both splits in the main run**, because the success rule cannot be met otherwise; their tests are uncorrected.

## 3. Confirmation design
Seeds 8650–8684 (35), splits A and B, 3 random label sets per seed, budgets 10/20/40/60, identical label sets for all learners (`nd_core.py`); unit = per-seed mean difference over budgets and draws; two-sided Wilcoxon signed-rank over seeds; bootstrap 95 % interval. Command: `bash scripts_rp/run_rp.sh confirm1 lap3_k10_st4,aw8_all_st2lr0.5,rp_gs4_his,rp_i4,rp_s3,rp_mc 8650-8684 A,B 4`.

## 4. Tests and success rule
- **Main family (Holm over 16)**: candidate − `lap3_k10_st4`, 4 candidates x 2 splits x {AUC, accuracy}.
- A candidate is **confirmed** iff (i) all four of its differences are positive, (ii) at least 3 of its 4 tests have Holm p < 0.05, (iii) it beats its shuffled-graph control and its capacity control (positive and uncorrected p < 0.05 for AUC or accuracy in at least one split, for each control). Otherwise "not confirmed".
- **HTR secondary family (exploratory, Holm within the family of 16)**: candidate − reference on HTR accuracy and HTR log-loss. Also reported without a verdict: log-loss, candidate − `aw8_all_st2lr0.5`, reference − `aw8_all_st2lr0.5` (does the learner result of the previous study replicate on new seeds?), per-type accuracy, per-budget gains.
- Power note (development): the per-seed sd of the paired differences is 0.007–0.017 AUC and 0.012–0.023 accuracy, so with 35 seeds the smallest effect that can be told from zero after Holm is about 0.005–0.01.
- "Not confirmed" is not "no effect": the intervals are reported.

## 5. Stated limits (written before the run)
Same 168 pair groups as all earlier work (new seeds are re-splits, not new data), so p-values are optimistic; one head schedule (not re-tuned per representation); the only second encoder is a cached ImageNet-ResNet-18 feature table (no augmentation of it possible); development saw no representation improving AUC and accuracy together, so a negative confirmation is the expected outcome.

# Pre-registration RT1 (red team) — written before any confirmation seed (8850–8884) was run

Written 2026-10-13 after the development audits on seeds 8800–8819 (a1 novelty / anchor-proximity stratification, a1c anchor-weight check, a6 identity audit, a7 aggregation sensitivity; a2, a3, a4, a5 were still running). Code: `new_directions/rt_*.py`; tests: `graph_exploration/rt_confirm.py`.

## What the development audits showed (observations, not evidence)
1. **Data geometry.** The 168 labelled groups involve 284 frames of three growth runs. Test images are near-duplicates of training frames (median distance 1−cos to the nearest training image ≈ 0.0002–0.02 on the graph's centred, normalised features; the distance between different-type reference images has median 0.58, 5 % quantile 0.14). Strict de-duplication of the labelled data is impossible (at 1−cos ≤ 0.002 only 13–22 % of the candidate pool and 4 of ~30 initial judgments survive).
2. **Anchor proximity.** ≈ 39 % of held-out judgments involve an image within 1−cos ≤ 0.001 of a typed reference image, ≈ 47 % of any typed anchor (references, utility and outer-test ideal images), ≈ 64 % within 0.01. Typed ideal images are therefore (near-)copies of trajectory frames.
3. **Where the learner gains are.** AUC gain of `lap3_k10_st4` over `aw8_all_st2lr0.5`/`aw8_all` is concentrated in the anchor-near stratum (≤ 0.001: +0.03–0.04); in judgments far (> 0.01) from every anchor it is small with intervals that include 0 (AUC +0.005…+0.009 vs `aw8_all`); the 4× schedule adds nothing there (−0.006…+0.003). In contrast the anchor-weight gain (`aw8_all` vs the default head) is largest in anchor-far judgments (+0.035…+0.044 AUC), so that earlier finding is not an anchor-copy artefact.
4. **Novelty to the training images** does not explain the gain (gain positive in all four novelty quartiles).
5. **Identity audit.** Graph nodes, labelled/pool images and anchors never share a path with a held-out image; by SHA-256 content hash one labelled/pool image has the same content as a held-out image in 3 of 10 audited cells (e.g. `115_RR220204A_441C_1738.bmp` vs `114_RR220204A_440C_1738.bmp`); `aw8_all` anchors by design include the 28 outer-test ideal images (harmless for the paired comparison, invalid as an external endpoint).
6. **Independent re-implementation** reproduces the main implementation's predictions to ≤ 1e-15 for all four learners on the audited cell (the Laplacian edge-sampling generator seed was copied; graph construction and identity filtering are independent).
7. **Aggregation sensitivity (stored confirmations).** The `lap3_k10_st4` gains over `aw8_all_st2lr0.5` and `lap3_k10` stay positive under budget-subset/median aggregations and sign-flip tests; the earlier `lap3_k10` vs `aw8_all` accuracy gain of graph-learner confirmation 2 is weak under several aggregations.

## Confirmatory tests (seeds 8850–8884, 35 seeds, splits A and B; one family, Holm over 18 tests)
Candidate learner `lap3_k10_st4`, reference `aw8_all_st2lr0.5` (best graph-free schedule of the earlier study). Per seed the gain is the mean over budgets 10/20/40/60 and 3 random label sets of the paired difference on identical label sets; two-sided Wilcoxon over seeds; paired bootstrap CI.
- `all` (standard endpoint) × {A, B} × {AUC, accuracy}: 4 tests (replication).
- `far`: only judgments whose two images are both > 0.01 (cosine distance) from every typed anchor (`anc_all_min` of `rt_a1b.py`) × {A, B} × {AUC, accuracy}: 4 tests.
- `dedup`: graph nodes within cosine distance 0.003 of any held-out image removed (labelled data unchanged) × {A, B} × {AUC, accuracy}: 4 tests.
- `blocked`: time-blocked contexts (100-frame blocks of a run, 25-frame buffer; training groups, pool and graph nodes keep no image within the buffer of a test block; test groups = groups with ≥ 1 image in a test block; `rt_blocked.py`) × {AUC, accuracy}: 2 tests.
- `aw8far` (anchor-weight check): `aw8_all` vs the default head `baseline` on `far` judgments × {A, B} × {AUC, accuracy}: 4 tests.
Verdict per evaluation: **robust** = all its gains positive and ≥ 3 of 4 (2 of 2 for `blocked`) Holm p < 0.05; **weakened** = all positive but fewer significant; **not supported** = some gain ≤ 0. For `far`, "weakened/not supported" is reported as "the confirmed learner gain is mostly anchor-near". No re-tuning after the results; the conclusions of earlier studies are annotated, not edited.

## Runs
```
export PYTHONPATH=code/active_learning_program
bash scripts_rt/run_rt.sh rt_a1.py c_a1 --seeds 8850-8884 --splits A,B --learners baseline,aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4
python3 rt_a1b.py --seeds 8850-8884 --splits A,B --out results/rt/c_a1b
bash scripts_rt/run_rt.sh rt_a3.py c_a3 --seeds 8850-8884 --learners aw8_all_st2lr0.5,lap3_k10_st4,lap3_k10_st4_shuf
bash scripts_rt/run_rt.sh rt_a2.py c_a2 --seeds 8850-8884 --splits A,B --deltas 0,0.003
python3 rt_confirm.py ../../new_directions/results/rt/c_a1 ../../new_directions/results/rt/c_a1b ../../new_directions/results/rt/c_a3 ../../new_directions/results/rt/c_a2
```
Limits: same 168 groups; a 20-seed development audit chose the thresholds (0.01 for `far`, 0.003 for `dedup`, 100/25 frames for `blocked`); `rt_a1b` distances use the repository's cached SimCLR features (the same geometry as the graph); unseen sessions/runs cannot be evaluated (only three runs exist).

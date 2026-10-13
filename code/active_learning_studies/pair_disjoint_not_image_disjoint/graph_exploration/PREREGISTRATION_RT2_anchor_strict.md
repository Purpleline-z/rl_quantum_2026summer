# Pre-registration RT2 (addendum to RT1): anchor-strict evaluation — written before any confirmation cell of it exists

Motivation (development audits 1 and 9, seeds 8800–8819 / 8800–8809): about 39 % of held-out judgments involve an image within cosine distance 0.001 of a typed ideal image, i.e. typed anchors are near-copies of held-out trajectory frames, and the learner gain of `lap3_k10_st4` is concentrated there. The cleanest test of "does the gain survive when the typed labels of copies of held-out frames are not available?" is to remove such anchors for all learners alike. Development run (10 seeds, budgets 20/40, 2 draws, ≈ 23 % of the 144 anchors removed): `lap3_k10_st4` − `aw8_all_st2lr0.5` AUC +0.030 (A) / +0.026 (B), accuracy +0.040 / +0.037; `aw8_all` − default head: AUC −0.022 (A) / +0.009 (B) (anchor weight 8 no longer helps when the copy-anchors are gone).

**Evaluation.** For every seed/split remove from the references, utility images and outer-test ideal images every ideal image with cosine distance ≤ 0.003 to any held-out (validation/test) image; everything else unchanged (`rt_a9.py`; learners `baseline`, `aw8_all`, `aw8_all_st2lr0.5`, `lap3_k10`, `lap3_k10_st4`; budgets 20 and 40; 2 random label sets; identical label sets for all learners). Seeds 8850–8884 (the RT1 confirmation seeds; this is a second, separately registered evaluation on them, run once), splits A and B.
**Tests (one family, Holm over 8):** (i) `lap3_k10_st4` − `aw8_all_st2lr0.5`, (ii) `aw8_all` − `baseline` (anchor-weight claim), each × {A, B} × {AUC, accuracy}; per-seed mean gain, two-sided Wilcoxon, paired bootstrap CI.
**Verdict rule** as RT1: robust = all four gains of the evaluation positive and ≥ 3 Holm p < 0.05; weakened = all positive, fewer significant; not supported = some gain ≤ 0.
**Stated expectation:** (i) robust; (ii) not supported (dev: negative in A).
```
bash scripts_rt/run_rt.sh rt_a9.py c_a9 --seeds 8850-8884 --splits A,B
python3 rt_report.py   # table name confirm_a9
```

# Progress log — graph-endpoint study (branch `claude/graph-endpoint-1013`)
Times are container UTC, 2026-10-13. Dev observations are not evidence; only blocks 8500–8534 (conf1) and 8550–8584 (conf2) count.

- 03:02 start; fetched, branch from `claude/graph-learner-20261012`; classification-baseline numbers read from `claude/classification-baselines` (read-only). Built the all-image SimCLR feature cache (1278 images; `nd_temporal.py build`).
- 03:20 own seed block in `nd_core.py` (dev 8400–8419, conf 8500–8534, 8550–8584; assertions). NOTE: the sibling selection session had used 8000–8479 (its S3 blocks 8400–8479), so the dev block overlaps its seeds 8400–8419 (different question; no dev number reused). Confirmation blocks are unused anywhere.
- 03:30 classification module (`ge_classify.py`): 1-NN control reproduces the baseline branch's five paper-seed numbers exactly (0.857, 0.857, 0.893, 0.821, 0.929); logistic regression 0.857 mean (their table: 0.850). Identity safety tests.
- 03:45 dev screens (seeds 8400–8419): classification — no graph method beats 1-NN (0.891) or logistic regression (0.907); best graph method (Laplacian-regularised logistic regression) = logistic regression. Dose-response — step then plateau/decline. Harsh splits — lap3_k10 keeps about +0.02 AUC. lambda-by-validation — not better than fixed lambda 3.
- 04:19 PREREGISTRATION_GE1 committed (before any seed >= 8500). 04:20 confirmation chain started (class -> harsh -> dose for conf1, then conf2).
- 04:42 conf1 classification done: no graph method beats 1-NN; lap3_ref (the Laplacian head) is significantly WORSE (-0.044, Holm 0.007).

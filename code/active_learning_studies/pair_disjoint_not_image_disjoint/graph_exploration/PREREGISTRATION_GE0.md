# PREREGISTRATION_GE0 — development plan (graph-endpoint study; written 2026-10-13 before any development result)
Branch `claude/graph-endpoint-1013` (from `claude/graph-learner-20261012`). Seeds: dev 8400–8419, confirmation 1 = 8500–8534, confirmation 2 = 8550–8584 (own block in `nd_core.py`; the sibling selection session used 8000–8479, so dev seeds 8400–8419 overlap its seeds — different question, no dev number is reused; confirmation seeds are unused everywhere).

## Protocol extensions (all marked)
E1 classification endpoint: type accuracy of the 28 outer-test ideal images on the paper's exact per-seed split (references 88 labelled, utility-validation 28 never labelled, outer test 28), instead of held-out preference prediction. Labelled data for pure classifiers = references only (as the classification-baseline branch). Two graph protocols reported separately: INDUCTIVE (test images not nodes, attach to nearest k nodes) and TRANSDUCTIVE (test images are unlabelled nodes; looser). Controls NN and logistic regression re-implemented from the baseline branch (read-only): they reproduce its 1-NN numbers on the five paper seeds exactly (0.857, 0.857, 0.893, 0.821, 0.929).
E2 unlabelled-image dose-response: graph nodes = ideal images (always) + a fixed per-seed random fraction p in {0, .25, .5, 1} of the other allowed trajectory images.
E3 harsh preference splits: (a) strict image-disjoint — the existing protocol ALREADY enforces it (no evaluation image occurs in any training/candidate group; verified by an assertion in `Context`); (b) held-out growth run (leave-one-run-out; run id parsed from the file name, 3 runs, parse success 284/284 images) and held-out temperature tercile (top third); held-out images (all frames of the held-out run/condition) are excluded from the graph (inductive). Groups spanning a held-out and a training condition are dropped.
E4 lambda selection by internal validation.

## Development variants (<= 10, dev seeds 8400–8419)
Classification (inductive and transductive each): `LS` label spreading from the references; `SGC` (logistic regression on propagated features); `GCN` (2-layer node classifier, cross-entropy on references); `lap3` (lap3_k10 head with reference-only anchors, aw 8, win-rate rule) vs `aw8` head (reference-only anchors) vs the NN/logreg controls; `fusion` (lap head win-rates + label-spreading posterior). Shuffled-graph control for LS/GCN. Pair-trained heads at budgets 0 and 60 on random label sets.
Dose-response: lap3_k10 minus aw8_all at p = 0, .25, .5, 1 (series 'all trajectory frames' and series 'frames outside every pair group'); shuffled control at p = 1.
Harsh splits: lap3_k10, aw8_all, lap3_k10_shuf, lap3_k10_n3 on held-out run and held-out temperature.
Lambda: validation-selected lambda in {1,3,10} vs fixed 3.
Freeze rule: <= 4 candidates (controls count as comparator arms, not candidates) and their families written to `PREREGISTRATION_GE1.md` and committed before any seed >= 8500 runs.

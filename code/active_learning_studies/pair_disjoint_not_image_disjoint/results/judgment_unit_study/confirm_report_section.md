### 5.15 Confirmation of the type-aware graph acquisition rules on unseen seeds

**Design.** Three rules were frozen on development seeds 400-409 before any confirmation seed was run (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`; the freezing is recorded in `graph_exploration/AUTONOMOUS_RUN_LOG.md`). They were then run once on 25 seeds not used for design (410-429 and 42, 79, 123, 202, 303), under the protocol of §5.14 (query unit = one (pair, type) judgment; budgets 10, 20, 40 and 60 judgments; splits A and B; single-shot and sequential). Gains are per-seed differences from the mean of five Random draws, averaged over the four budgets; the p-value is a seed-level Wilcoxon signed-rank test, Holm-corrected over the three frozen candidates within each block; controls and embedding baselines are shown with raw p-values and are not part of the Holm family. The success criterion fixed in advance was: Holm p < 0.05 for log-loss in at least two of the four blocks including both splits, with a non-negative AUC gain in those blocks. The candidates were chosen because they looked best on the development seeds, whose per-seed standard deviation of the gain is 0.09-0.14, so development gains are optimistic estimates.

**Rules.** The graph rules build a ten-nearest-neighbour graph over the images of the candidate and revealed judgments and the typed reference images, spread the reference types over it (label spreading), predict the probability that a judgment is decisive from the resulting type-posterior features and the revealed outcomes, and select by farthest-first coverage in graph-propagated pair space weighted by that probability (and by own-head uncertainty for the primary rule); the two controls replace the decisive predictor by a type-only one or shuffle the posterior (§5.14 and `graph_exploration/DESIGN_AND_LITERATURE.md`).

| Split | Condition | Rule | log-loss gain | p (Holm for candidates) | AUC gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) x uncertainty | -0.002 | 1.000 | +0.0068 | 48% | 25 |
| A | single-shot | Type-aware graph coverage x P(decisive) | -0.007 | 1.000 | -0.0025 | 56% | 25 |
| A | single-shot | Laplace BALD x P(decisive), graph features | +0.007 | 1.000 | +0.0061 | 60% | 25 |
| A | single-shot | (control) as first, decisive predictor from type only | +0.026 | 0.191 (raw) | -0.0003 | 60% | 25 |
| A | single-shot | (control) as first, type posterior shuffled over images | +0.040 | 0.059 (raw) | +0.0026 | 64% | 25 |
| A | single-shot | (baseline) Core-set, relation-aware pairs | +0.042 | 0.191 (raw) | -0.0042 | 60% | 25 |
| A | single-shot | (baseline) BALD x P(decisive), pair-distance features | +0.007 | 0.672 (raw) | +0.0058 | 56% | 25 |
| A | single-shot | (baseline) TypiClust (pairs) | -0.024 | 0.771 (raw) | -0.0055 | 60% | 25 |
| A | single-shot | (baseline) Laplace BALD | -0.011 | 0.853 (raw) | +0.0022 | 48% | 25 |
| A | single-shot | (baseline) Uncertainty, own head | +0.011 | 0.692 (raw) | +0.0013 | 56% | 25 |
| A | sequential | Type-aware graph coverage x P(decisive) x uncertainty | -0.008 | 1.000 | +0.0056 | 60% | 25 |
| A | sequential | Type-aware graph coverage x P(decisive) | -0.010 | 1.000 | -0.0007 | 48% | 25 |
| A | sequential | Laplace BALD x P(decisive), graph features | -0.006 | 1.000 | +0.0057 | 56% | 25 |
| A | sequential | (control) as first, decisive predictor from type only | +0.030 | 0.067 (raw) | +0.0071 | 64% | 25 |
| A | sequential | (control) as first, type posterior shuffled over images | +0.006 | 0.653 (raw) | +0.0040 | 52% | 25 |
| A | sequential | (baseline) Core-set, relation-aware pairs | +0.050 | 0.059 (raw) | +0.0018 | 64% | 25 |
| A | sequential | (baseline) BALD x P(decisive), pair-distance features | +0.001 | 0.791 (raw) | +0.0040 | 56% | 25 |
| A | sequential | (baseline) TypiClust (pairs) | -0.005 | 0.812 (raw) | +0.0028 | 52% | 25 |
| A | sequential | (baseline) Laplace BALD | -0.025 | 0.979 (raw) | +0.0028 | 56% | 25 |
| A | sequential | (baseline) Uncertainty, own head | -0.012 | 0.958 (raw) | +0.0027 | 44% | 25 |
| B | single-shot | Type-aware graph coverage x P(decisive) x uncertainty | +0.002 | 0.895 | +0.0049 | 48% | 25 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.062 | 0.014 | +0.0130 | 76% | 25 |
| B | single-shot | Laplace BALD x P(decisive), graph features | +0.019 | 0.625 | +0.0061 | 60% | 25 |
| B | single-shot | (control) as first, decisive predictor from type only | -0.005 | 0.672 (raw) | -0.0031 | 44% | 25 |
| B | single-shot | (control) as first, type posterior shuffled over images | +0.031 | 0.067 (raw) | +0.0031 | 64% | 25 |
| B | single-shot | (baseline) Core-set, relation-aware pairs | +0.006 | 0.525 (raw) | -0.0053 | 60% | 25 |
| B | single-shot | (baseline) BALD x P(decisive), pair-distance features | +0.015 | 0.542 (raw) | +0.0055 | 60% | 25 |
| B | single-shot | (baseline) TypiClust (pairs) | -0.014 | 0.895 (raw) | -0.0033 | 56% | 25 |
| B | single-shot | (baseline) Laplace BALD | -0.005 | 0.937 (raw) | +0.0034 | 52% | 25 |
| B | single-shot | (baseline) Uncertainty, own head | -0.046 | 0.055 (raw) | -0.0145 | 36% | 25 |
| B | sequential | Type-aware graph coverage x P(decisive) x uncertainty | +0.050 | 0.055 | +0.0146 | 72% | 25 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.083 | 0.002 | +0.0185 | 80% | 25 |
| B | sequential | Laplace BALD x P(decisive), graph features | +0.049 | 0.055 | +0.0151 | 76% | 25 |
| B | sequential | (control) as first, decisive predictor from type only | +0.040 | 0.127 (raw) | +0.0036 | 72% | 25 |
| B | sequential | (control) as first, type posterior shuffled over images | +0.081 | <0.001 (raw) | +0.0148 | 84% | 25 |
| B | sequential | (baseline) Core-set, relation-aware pairs | +0.063 | 0.034 (raw) | +0.0056 | 72% | 25 |
| B | sequential | (baseline) BALD x P(decisive), pair-distance features | +0.055 | 0.101 (raw) | +0.0102 | 64% | 25 |
| B | sequential | (baseline) TypiClust (pairs) | +0.035 | 0.182 (raw) | +0.0079 | 56% | 25 |
| B | sequential | (baseline) Laplace BALD | +0.024 | 0.275 (raw) | +0.0061 | 60% | 25 |
| B | sequential | (baseline) Uncertainty, own head | +0.012 | 0.711 (raw) | +0.0020 | 48% | 25 |

**Verdict against the pre-registered criterion.**

```
Success criterion (Holm p < 0.05 for log-loss in >= 2 blocks incl. both splits, AUC gain >= 0 there):
  typed_decisive_coverage_unc: significant blocks [] -> not met
  typed_decisive_coverage: significant blocks [('B', 'single'), ('B', 'sequential')] -> not met
  typed_decisive_bald: significant blocks [] -> not met
```

**Result.** The pre-registered criterion was not met: no frozen candidate is significantly better than Random in both splits. The plain rule `typed_decisive_coverage` is significantly better than Random in split B in both conditions (log-loss gain +0.062 single-shot and +0.083 sequential, Holm p = 0.014 and 0.002; AUC +0.013 and +0.019, seeds better 76% and 80%), and it has the largest gain among all rules in the table in those two blocks; in split A its gains are −0.007 and −0.010 (Holm p = 1.0). The primary rule `typed_decisive_coverage_unc` and `typed_decisive_bald` are not significant after correction in any block (the sequential split-B gains of +0.050 and +0.049 have Holm p = 0.055). Among the embedding baselines (raw p-values, not corrected for multiplicity) Core-set with relation features has +0.042 / +0.050 in split A and +0.006 / +0.063 in split B, and BALD × P(decisive) with pair-distance features +0.007 / +0.001 and +0.015 / +0.055; none of the baselines has a raw p-value below 0.05 in both splits. The Uncertainty rule is below Random in split B single-shot (−0.046).

**The graph is not shown to be responsible.** For the primary rule the two controls were run on the same seeds. Replacing the type posterior by a shuffled one gives +0.040 / +0.006 (split A) and +0.031 / +0.081 (split B), and replacing the decisive predictor by one that uses only the type gives +0.026 / +0.030 and −0.005 / +0.040; the difference between the rule and its shuffled control is −0.042, −0.013, −0.029 and −0.031 log-loss in the four blocks (raw p between 0.07 and 0.83), i.e. the real posterior is never better than the shuffled one. In split B the sequential gain of the shuffled-posterior control (+0.081, raw p < 0.001) is larger than that of the real rule (+0.050). The controls were run only for the primary rule, so for the plain rule that is significant in split B we cannot say how much of its gain comes from the graph and how much from coverage in a smoothed feature space. The earlier first-generation result (§5.14) also found that the graph-propagated core-set equals its shuffled control. What is consistent across §5.13, §5.14 and this section is that coverage-type rules help in the classifier2-style split B (larger candidate pool) and not reliably in the small-pool split A; this section does not provide evidence that graph structure adds to that.

**Why the development gains did not carry over.** On the development seeds (400-409) the primary rule gained +0.108 / +0.138 / +0.029 / +0.091 log-loss over Random. On the 25 unseen seeds it gains −0.002 / −0.008 / +0.002 / +0.050. On the development seeds Random's own log-loss was unusually poor (0.648 against 0.548 on the confirmation seeds in split A, sequential) while the rule's log-loss was similar on both sets (0.510 and 0.554), and about 20 variants had been compared on those 10 seeds, so the development gain was an optimistic estimate of an effect that is small or absent. With a per-seed standard deviation of the gain of about 0.1, 25 seeds detect gains of roughly 0.04 or more; smaller true effects cannot be excluded.

**Limits.** One encoder (SimCLR features of one set of three sessions), one value of k and one propagation weight, a linear decisive predictor fitted to 30-100 revealed judgments, selectors that see the typed reference images, and a graph restricted to the images of candidates, revealed judgments and references (no unlabelled trajectory frames, no temporal edges). A trained graph network, a second encoder and the unlabelled trajectory images remain untested.


**Replication on fresh seeds (split B only).** Because the plain rule was significant in split B but its own controls had not been run, a replication was pre-registered in `graph_exploration/AUTONOMOUS_RUN_LOG.md` before it ran: seeds 430-459 (30 seeds never used before), split B, both conditions, with no change to the rule, plus its type-only and shuffled-posterior controls and the Core-set baseline. Gains are over Random as above; p-values are raw seed-level Wilcoxon tests (the paired differences are not corrected for multiplicity; H1 is corrected over the two conditions).

| Condition | Rule | log-loss gain | raw p | AUC gain | seeds better |
|---|---|---:|---:|---:|---:|
| single-shot | Type-aware graph coverage x P(decisive) | +0.042 | 0.029 | +0.0078 | 67% (30) |
| single-shot | (control) decisive predictor from the type only | +0.038 | 0.038 | +0.0043 | 70% (30) |
| single-shot | (control) type posterior shuffled over images | +0.060 | <0.001 | +0.0093 | 80% (30) |
| single-shot | (baseline) Core-set, relation-aware pairs | -0.015 | 0.715 | -0.0008 | 50% (30) |
| sequential | Type-aware graph coverage x P(decisive) | +0.030 | 0.393 | +0.0071 | 50% (30) |
| sequential | (control) decisive predictor from the type only | +0.038 | 0.158 | +0.0008 | 60% (30) |
| sequential | (control) type posterior shuffled over images | +0.034 | 0.221 | +0.0024 | 63% (30) |
| sequential | (baseline) Core-set, relation-aware pairs | -0.019 | 0.503 | -0.0093 | 50% (30) |

| Condition | Paired difference (log-loss) | mean | raw p |
|---|---|---:|---:|
| single-shot | rule minus type-only control | +0.004 | 0.968 |
| single-shot | rule minus shuffled-posterior control | -0.018 | 0.529 |
| single-shot | rule minus Core-set baseline | +0.057 | 0.058 |
| sequential | rule minus type-only control | -0.008 | 0.440 |
| sequential | rule minus shuffled-posterior control | -0.003 | 0.919 |
| sequential | rule minus Core-set baseline | +0.049 | 0.124 |

H1 (gain over Random, Holm over the two conditions): single-shot p = 0.059, sequential p = 0.393.

In single-shot selection the plain rule again has a positive gain over Random (+0.042 log-loss, raw p = 0.029, Holm p = 0.059 over the two conditions, AUC +0.008, 67% of seeds better), smaller than in the confirmation (+0.062); in sequential selection the gain is +0.030 (Holm p = 0.39). The replication therefore does not reach significance after correction, and the point estimates are about half of those of the confirmation, as expected when a rule has been selected for a good result. The controls are as good as the rule: using only the type to predict decisiveness gives +0.038 in both conditions, shuffling the type posterior gives +0.060 and +0.034, and the paired differences between the rule and its controls are +0.004 / −0.008 (type-only) and −0.018 / −0.003 (shuffled) log-loss, none distinguishable from zero. The Core-set baseline, which gained +0.063 in the confirmation, has −0.015 and −0.019 here; the rule minus Core-set difference is +0.057 (raw p = 0.058) and +0.049 (raw p = 0.12). The supported statement is therefore narrow: weighting a coverage rule by a predicted probability that the judgment will be decisive, even when that probability uses only the reconstruction type, gives a small gain of about 0.03-0.06 log-loss over Random in the classifier2-style split B, at the edge of detectability with 25-30 seeds; the graph-derived type posterior does not add to it in these data, and the effect is not present in the small-pool split A.


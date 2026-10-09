### 5.15 Confirmation of the type-aware graph acquisition rules on unseen seeds

**Design.** Three rules were frozen on development seeds 400-409 before any confirmation seed was run (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`; the freezing is recorded in `graph_exploration/AUTONOMOUS_RUN_LOG.md`). They were then run once on 25 seeds not used for design (410-429 and 42, 79, 123, 202, 303), under the protocol of §5.14 (query unit = one (pair, type) judgment; budgets 10, 20, 40 and 60 judgments; splits A and B; single-shot and sequential). Gains are per-seed differences from the mean of five Random draws, averaged over the four budgets; the p-value is a seed-level Wilcoxon signed-rank test, Holm-corrected over the three frozen candidates within each block; controls and embedding baselines are shown with raw p-values and are not part of the Holm family. The success criterion fixed in advance was: Holm p < 0.05 for log-loss in at least two of the four blocks including both splits, with a non-negative AUC gain in those blocks. The candidates were chosen because they looked best on the development seeds, whose per-seed standard deviation of the gain is 0.09-0.14, so development gains are optimistic estimates.

**Rules.** The graph rules build a ten-nearest-neighbour graph over the images of the candidate and revealed judgments and the typed reference images, spread the reference types over it (label spreading), predict the probability that a judgment is decisive from the resulting type-posterior features and the revealed outcomes, and select by farthest-first coverage in graph-propagated pair space weighted by that probability (and by own-head uncertainty for the primary rule); the two controls replace the decisive predictor by a type-only one or shuffle the posterior (§5.14 and `graph_exploration/DESIGN_AND_LITERATURE.md`).

| Split | Condition | Rule | log-loss gain | p (Holm for candidates) | AUC gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) x uncertainty | +0.002 | 1.000 | +0.0094 | 44% | 9 |
| A | single-shot | Type-aware graph coverage x P(decisive) | -0.027 | 1.000 | +0.0007 | 44% | 9 |
| A | single-shot | Laplace BALD x P(decisive), graph features | -0.007 | 1.000 | +0.0090 | 60% | 10 |
| A | single-shot | (control) as first, decisive predictor from type only | -0.015 | 0.734 (raw) | -0.0040 | 44% | 9 |
| A | single-shot | (control) as first, type posterior shuffled over images | +0.051 | 0.129 (raw) | +0.0113 | 78% | 9 |
| A | single-shot | (baseline) Uncertainty, own head | +0.011 | 0.692 (raw) | +0.0013 | 56% | 25 |
| A | sequential | Type-aware graph coverage x P(decisive) x uncertainty | +0.029 | 1.000 | +0.0150 | 78% | 9 |
| A | sequential | Type-aware graph coverage x P(decisive) | -0.018 | 1.000 | -0.0001 | 44% | 9 |
| A | sequential | Laplace BALD x P(decisive), graph features | -0.036 | 1.000 | +0.0066 | 56% | 9 |
| A | sequential | (control) as first, decisive predictor from type only | +0.045 | 0.301 (raw) | +0.0132 | 56% | 9 |
| A | sequential | (control) as first, type posterior shuffled over images | +0.018 | 0.734 (raw) | +0.0087 | 56% | 9 |
| A | sequential | (baseline) Uncertainty, own head | -0.012 | 0.958 (raw) | +0.0027 | 44% | 25 |
| B | single-shot | Type-aware graph coverage x P(decisive) x uncertainty | -0.004 | 1.000 | -0.0064 | 50% | 6 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.101 | 0.047 | +0.0213 | 100% | 7 |
| B | single-shot | Laplace BALD x P(decisive), graph features | +0.054 | 0.062 | +0.0023 | 86% | 7 |
| B | single-shot | (control) as first, decisive predictor from type only | -0.031 | 0.438 (raw) | -0.0097 | 33% | 6 |
| B | single-shot | (control) as first, type posterior shuffled over images | +0.040 | 0.438 (raw) | -0.0036 | 67% | 6 |
| B | single-shot | (baseline) Uncertainty, own head | -0.046 | 0.055 (raw) | -0.0145 | 36% | 25 |
| B | sequential | Type-aware graph coverage x P(decisive) x uncertainty | +0.075 | 0.625 | +0.0130 | 67% | 6 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.098 | 0.469 | +0.0197 | 83% | 6 |
| B | sequential | Laplace BALD x P(decisive), graph features | +0.012 | 0.688 | +0.0013 | 67% | 6 |
| B | sequential | (control) as first, decisive predictor from type only | -0.040 | 0.562 (raw) | -0.0156 | 33% | 6 |
| B | sequential | (control) as first, type posterior shuffled over images | +0.085 | 0.219 (raw) | +0.0168 | 83% | 6 |
| B | sequential | (baseline) Uncertainty, own head | +0.012 | 0.711 (raw) | +0.0020 | 48% | 25 |

**Verdict against the pre-registered criterion.**

```
Success criterion (Holm p < 0.05 for log-loss in >= 2 blocks incl. both splits, AUC gain >= 0 there):
  typed_decisive_coverage_unc: significant blocks [] -> not met
  typed_decisive_coverage: significant blocks [('B', 'single')] -> not met
  typed_decisive_bald: significant blocks [] -> not met
```

(Interpretation paragraph not yet written.)


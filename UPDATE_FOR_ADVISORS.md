# Project update: choosing which pairwise RHEED judgments to label

**Setting.** A Bradley–Terry reward model (ResNet-18, SimCLR-initialised, one reward head per reconstruction type) is trained from expert pairwise judgments of STO RHEED images. We ask which pair groups should be sent to the experts next, at labelling budgets of 10–60 pair groups. The usable data are 168 pair groups (521 judgments); 51% of judgments are decisive, 14% ties, 35% "not applicable". Evaluation images never appear in training (SHA-256 content identity).

## What we did
1. **Merged and extended the earlier seed 202/303 work** so the eight-strategy comparison covers five seeds, and checked the low accuracies (0.4–0.7) you noticed. We found no implementation bug; the causes are design-related: very few labelled rows for end-to-end fine-tuning, training that changes with the *order* of the selected pairs, at budget 100 all strategies receiving the identical 100 groups (the candidate pool had only 100), and a 28-image test set. Rankings that looked clear on five seeds (e.g. Cluster-Margin, core-set) did not survive more seeds.
2. **Froze the SimCLR encoder** and trained only the reward head in a way that does not depend on the order of the pairs. This removes most training noise (run-to-run s.d. about 0.04).
3. **Found that ideal-image type accuracy cannot compare strategies.** The reference-image anchors alone give 0.848 accuracy with no pair labels at all; pair labels alone give 0.37–0.55; a frozen 1-nearest-neighbour baseline gives 0.871 ± 0.041. The pair labels teach *preference*, not class identity.
4. **Switched to a held-out preference endpoint** (log-loss and AUC on the decisive judgments of held-out pair groups, image-disjoint from training) and compared 33 strategy variants — the original eight, 17 new ones from the literature (BADGE, TypiClust, Fisher/D-optimal design as in Active Reward Modeling, Laplace-approximation BALD, DPP, ProbCover, FASS, graph cut, deep-ensemble BALD, …) and "all-head" versions of the originals — against random selection, on **35 random splits**, in two modes (one batch chosen at once; sequential rounds of 10), with Wilcoxon tests and Holm correction.
5. Repeated the comparison with the **data split of the earlier classifier2 work** (20% of pair groups held out).

## Main results
- **No strategy is robustly better than random selection.** Gains are small (log-loss about 0.03–0.05, AUC at most about 0.01, against a random baseline log-loss of about 0.36–0.58).
- **Which rule helps depends on the split.** With a small split (40 test groups, pool of 64–77): Laplace BALD and BALD × P(decisive) beat random when one batch is chosen (about −0.04 log-loss, +0.01 AUC), but not with sequential retraining. With the classifier2-style split (34 test groups, pool of 99–113): only plain core-set is significant (+0.052 single-shot, +0.053 sequential, log-loss only). No rule is significant under both splits.
- **Consistent across splits and modes:** uncertainty-driven selection (uncertainty sampling, MC-dropout, BADGE-style, DropQuery) is never significantly better than random; coverage/diversity rules tend to lead at the smallest budgets (10–20 groups), but the leader varies. Graph cut is significantly worse than random in the small split.
- A cold-start check (which first 10 groups to label) shows no detectable benefit from any choice over random, over 35 seeds.

## Caveats
- Test sets are small (35–79 decisive judgments per seed); Holm correction is applied within each cell, not across all budgets/metrics/conditions; per-budget "winners" are the best of 32 variants and overstate.
- Hyper-parameters were tuned once, on random batches from three seeds; a second setting changes which strategies are significant on log-loss, while AUC conclusions are more stable.
- The original strategies score the type of the group's first judgment (the queried type), but a selected group delivers all of its judgments (about 3.1 on average) while the budget counts groups.

## Questions for you
1. Is the real query unit a (pair, type) judgment? If so, we should count the budget in judgments and select (pair, type) rather than pair groups.
2. Is held-out preference prediction (log-loss/AUC) an acceptable primary endpoint, now that ideal-image type accuracy is saturated by the reference anchors?
3. Would you want a larger candidate pool and a second encoder (ImageNet or a larger self-supervised model) before any recommendation is made?

*Everything is in branch `claude/frozen-encoder-strategies` of `rl_quantum_2026summer`; the report is `code/ACADEMIC_REPORT_DRAFT.md` (§5.10–5.13), and the hand-off notes are in `code/active_learning_studies/pair_disjoint_not_image_disjoint/`.*

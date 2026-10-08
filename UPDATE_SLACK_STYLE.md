Hi Justin, update on the active selection results (what changed since the Oct 6 numbers). Figures attached: (1) the old 8-strategy accuracy curve, (2) the new strategy-vs-random plot.

1. What I saw when I re-checked the old 8-strategy results (fig 1)
   - With more seeds, the best strategy per budget changed.
   - Changing the order of the same training data changed the accuracy a lot.
   - At budget 100 every strategy got the same 100 pairs, but accuracy still varied a lot across strategies, which suggests it is training noise, not selection.
   - Test set is only 28 images; accuracy is 0.4–0.7.
   - I checked the code and didn't find a bug.
2. What I changed to reduce the noise
   - Froze the SimCLR encoder and trained only the reward head, in a way that doesn't depend on the order of the pairs.
   - Run-to-run s.d. went down to about 0.04 (same data, different training seeds).
3. Another observation about the endpoint (ideal-image type accuracy)
   - The ideal anchors alone give 0.848, pair labels alone give 0.37–0.55, a frozen 1-NN gives 0.871.
   - So this accuracy may not be sensitive to which pairs we label.
   - I switched to held-out preference log-loss / AUC on image-disjoint pair groups. It is the same quantity as classifier2's pairwise holdout accuracy, with two more metrics.
4. New comparison (fig 2: filled = significant after Holm)
   - 33 strategies: the original 8, 17 from the literature (BADGE, TypiClust, Fisher D-optimal, Laplace-BALD, DPP, ProbCover, ...), and all-head versions.
   - 35 seeds, one-shot and sequential rounds of 10, budgets 10–60.
   - Two splits: a small split (10 initial / 20 val / 40 test) and a classifier2-style 80/20 split.
5. What the results look like
   - We did not find a strategy that is consistently better than random; the gains are small (log-loss ~0.03–0.05, AUC < 0.01).
   - Which rule looks good depends on the split:
     - small split, one-shot: Laplace-BALD and BALD×P(decisive) are significant
     - classifier2 split: only core-set is significant
   - Uncertainty-based rules were never significantly better than random.
   - Coverage-type rules tend to be ahead at the smallest budgets, but the leader changes.
   - This does not agree with the Oct 6 table (uncertainty best at budget 100, Cluster-Margin at 25, etc.).
6. A flaw in my earlier design (budget and judgments)
   - The labeling GUI asks for one pair and one type per query, so the real query unit is one (pair, type) judgment.
   - My simulation picked a whole pair group and revealed all its judgments (~3.1 on average), with the budget counted in groups.
   - So "budget 10" was really ~31 judgments, and the other types came for free.
   - This may have hidden differences between strategies, so the "no strategy beats random" result may partly come from this setup.
7. Next steps
   - Rerunning everything with one judgment as the query unit, budget in judgments (10/20/40/60). Will update when it finishes.
8. Code and results
   - https://github.com/Purpleline-z/rl_quantum_2026summer/tree/claude/frozen-encoder-strategies
   - Report: code/ACADEMIC_REPORT_DRAFT.md §5.10–5.13
9. Questions
   - Is held-out preference log-loss / AUC OK as the main endpoint?
   - If the rerun still shows no gain over random, what should we do next (e.g. a larger pool, a second encoder, or label randomly)?

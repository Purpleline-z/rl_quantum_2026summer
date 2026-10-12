# Literature notes for Prompt 2 (graph selection under the tuned learner; 2026-10-12)
Only what the search results stated; "(summary)" = search summary, paper text not opened (arxiv/ar5iv blocked). Earlier notes (Sigma-optimality, Laplacian design, TypiClust/ProbCover, last-layer Laplace) are in `notes_overnight2_literature.md`, `notes_graph_literature_round.md` and `new_directions/literature/lit_selection.md` and are not repeated.

## Class-priority / imbalance-aware acquisition
- **Cold Start Active Learning Strategies in the Context of Imbalanced Classification (arXiv 2201.10227; summary).** Clustering + label propagation to choose what to label first; clusters ranked by a quality score plus an impurity score from label entropy; reported to raise recall of the minority class. Relevance: our HTR-oriented rules also spend budget where the propagated type posterior is impure (HTR vs RT13), but ours has no clustering step.
- **Learning on the Border: Active Learning in Imbalanced Data Classification (Ertekin et al., CIKM 2007; title/summary).** Near-boundary sampling produces more balanced label sets than random in imbalanced data. Relevance: motivates "bridge"/ambiguous-image selection as a way to oversample the rarer side.
- **DIRECT: Deep Active Learning under Imbalance and Label Noise (arXiv 2312.09196; summary).** Plain uncertainty sampling collects less balanced annotations; DIRECT annotates the most uncertain examples nearest an estimated separation threshold.
- **Class-Balanced and Reinforced Active Learning on Graphs (arXiv 2402.10074; summary).** Standard graph AL gives skewed label sets that hurt minority-class accuracy; learns an acquisition policy aiming at class-balanced informative nodes.
- **Exemplar Guided Active Learning (Hartford et al., NeurIPS 2020; summary).** Uncertainty-driven methods can exhaust the budget before a rare class is seen; uses embeddings to guide the search for rare-class examples.

## Boundary / bridge nodes in graph active learning
- **When Contrastive Learning Meets Active Learning: graph active learning with self-supervision (summary).** Warns of "propagation blindness": uncertainty tends to select nodes on community boundaries, which can add confusion; favours nodes inside homophilous subgraphs. This is evidence AGAINST pure "bridge" selection and is the prior for our expectation that bridge rules may not help.
- **Graph-Based Active Learning Based on Label Propagation (Long et al., 2008; summary).** Label-propagation classifier, pick the instance with the largest expected entropy reduction.
- **Information Gain Propagation (summary).** Score = expected information gain x influence magnitude; notes that high entropy is not the same as high information gain.
- Searches found no paper that selects bridge nodes between two classes by label-propagation posterior entropy; our HTR/RT13 rules are not a verified novelty claim.

## Abstaining annotators
- **Active learning from imperfect labelers (summary).** "I don't know" answers can be used as a signal that the decision boundary is near, and an abstention never misleads while a noisy label can. Relevance: our P(decisive) weight treats ties/not-applicable as wasted queries; this literature suggests they can be informative (about where types are ambiguous) — not tested here.

## Cold start
- **TypiClust / ProbCover (Hacohen et al. 2022; Yehuda et al. 2022; summaries).** At very low budgets uncertainty fails and diversity/coverage (dense typical points; max probability cover) works; TypiClust/ProbCover are the methods that consistently beat random at the first selection step (reviewer summary). Our `typed_decisive_coverage` is a farthest-first coverage rule, so cold start is where it should help most.

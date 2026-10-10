# Extra literature search: active preference learning (BT reward model on frozen encoder, 10-60 labels)

Method note: all entries were found via WebSearch result listings. WebFetch to arxiv.org failed (DNS error), so no paper text was read. Bibliographic details (title, authors, venue, year) come from search-result metadata (ACM DL, PMLR, OpenReview, arXiv, ML Anthology, Semantic Scholar). Claims about content are from search-result summaries. Anything not confirmed is marked UNVERIFIED.

## T13. Expected error reduction / look-ahead; expected model change / EGL

1. Roy, N. & McCallum, A. "Toward Optimal Active Learning through Sampling Estimation of Error Reduction." ICML 2001, pp. 441-448. https://dl.acm.org/doi/10.5555/645530.655646 (PDF: https://www.lri.fr/~sebag/Examens/Active/roy01toward.pdf). CONFIRMED.
   Applies: the original EER method. For each candidate, retrain on each possible label and estimate future error on the pool. For a BT head on frozen features, retraining is a tiny logistic regression, so exact look-ahead is cheap. Use fantasized labels weighted by current predictive probability.

2. Mohamadi, M. A., Bae, W., Sutherland, D. J. "Making Look-Ahead Active Learning Strategies Feasible with Neural Tangent Kernels." NeurIPS 2022. https://arxiv.org/abs/2206.12569 ; https://proceedings.neurips.cc/paper_files/paper/2022/hash/5132940b1bced8a7b28e9695d49d435a-Abstract-Conference.html . CONFIRMED.
   Applies: approximates retraining on hypothetical labels with the NTK (acquisition called MLMOC, "most likely model output change"). It is the efficient look-ahead variant and the closest modern analogue to fantasized-label retraining. With a frozen encoder the NTK is just the linear/feature kernel.

3. Freytag, A., Rodner, E., Denzler, J. "Selecting Influential Examples: Active Learning with Expected Model Output Changes." ECCV 2014 (LNCS 8692, pp. 562-577). https://link.springer.com/chapter/10.1007/978-3-319-10593-2_37 (PDF: https://www.erodner.de/pdf/Freytag14_SIE.pdf). CONFIRMED.
   Applies: expected model-output change marginalised over the unknown label, with efficient closed forms for GP regression. This is a direct template for a GP or linear-head variant. It is not for preferences.

4. Settles, B., Craven, M., Ray, S. "Multiple-Instance Active Learning." NIPS 2007 (proceedings volume sometimes cited as 2008). https://burrsettles.com/pub/settles.nips08.pdf . CONFIRMED (year ambiguity noted by sources).
   Applies: the standard citation for Expected Gradient Length. For a BT head the gradient norm of a pair is |y - sigma(r_a - r_b)| * ||phi_a - phi_b||. That is a cheap closed form to benchmark against.

Ranking/preference-specific (relevant, thinner):
- Donmez, P. & Carbonell, J. G. "Optimizing Estimated Loss Reduction for Active Sampling in Rank Learning." ICML 2008, pp. 248-255, DOI 10.1145/1390156.1390188. https://mlanthology.org/icml/2008/donmez2008icml-optimizing/ . CONFIRMED metadata. Estimated-loss-reduction (EER-style) sampling for rank learning with SVM/boosting rankers. Did not read content.
- Other candidates seen: Guo & Greiner, "Optimistic Active-Learning Using Mutual Information," IJCAI 2007, pp. 823-829 (https://www.ijcai.org/Proceedings/07/Papers/132.pdf), a look-ahead/optimistic EER relative. Mussmann et al., "Active Learning with Expected Error Reduction," arXiv 2211.09283 (https://arxiv.org/html/2211.09283v1), a Bayesian/MC-dropout EER that avoids retraining. Authors and venue UNVERIFIED beyond the arXiv ID. Zhao et al., "Direct Acquisition Optimization for Low-Budget Active Learning," arXiv 2402.06045 (https://arxiv.org/html/2402.06045v1), EER with influence functions aimed at low budgets. Venue and co-authors UNVERIFIED.
- No paper found applying EGL or expected model change specifically to pairwise preference or BT. That is a gap (UNVERIFIED that none exists).

Support: well supported for EER and EGL as methods. Thin for preference-specific applications.

## T14. Stochastic / randomised batch acquisition; BatchBALD; low-budget evidence

1. Kirsch, A., Farquhar, S., Atighehchian, P., Jesson, A., Branchaud-Charron, F., Gal, Y. "Stochastic Batch Acquisition: A Simple Baseline for Deep Active Learning." TMLR 2023 (arXiv 2106.12059, 2021). https://arxiv.org/abs/2106.12059 ; https://openreview.net/forum?id=vcHwQyNBjW . CONFIRMED.
   Applies: sample the batch from the pool with probability from the acquisition scores (softmax/Gibbs "softmax-BALD", also power and softmax-rank variants). It is O(M log K), the same as top-K, and competitive with BatchBALD and BADGE. Directly usable on pair scores to avoid redundant top-K pairs.

2. Kirsch, A., van Amersfoort, J., Gal, Y. "BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning." NeurIPS 2019. https://arxiv.org/abs/1906.08158 ; https://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning . CONFIRMED.
   Applies: joint mutual information of the batch (greedy, 1-1/e approximation). It shows top-K BALD picks near-duplicates and can be worse than random. This is the motivation for diversifying batches.

3. Hacohen, G., Dekel, A., Weinshall, D. "Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets." ICML 2022 (PMLR 162, pp. 8175-8195). https://arxiv.org/abs/2202.02794 ; https://proceedings.mlr.press/v162/hacohen22a.html . CONFIRMED.
   Applies: evidence that uncertainty-driven selection underperforms in the very-low-budget regime (representative/typical points work better, using self-supervised features). This supports adding randomness or representativeness at 10-60 labels. Counterpoint below.

4. Gupte, S. R., Aklilu, J., Nirschl, J. J., Yeung-Levy, S. "Revisiting Active Learning in the Era of Vision Foundation Models." TMLR 2024 (arXiv 2401.14555). https://arxiv.org/abs/2401.14555 ; https://openreview.net/forum?id=u8K83M9mbG . CONFIRMED metadata. Content details come from a secondary summary and are UNVERIFIED.
   Applies: with frozen foundation features and a linear head, uncertainty sampling is reportedly competitive from the start, contradicting the strict TypiClust picture. Its DropQuery recipe is representative init, then uncertainty shortlist, then diverse subset. Directly matches the setup here.

Thompson-style batching: Dai et al., "Sample-Then-Optimize Batch Neural Thompson Sampling," NeurIPS 2022 (https://arxiv.org/abs/2210.06850) uses independent posterior samples per batch slot. It is a bandit/BO setting, not active learning. Seen in search only. Authors beyond Dai UNVERIFIED. Batch preference: Biyik, E., Anari, N., Sadigh, D. "Batch Active Learning of Reward Functions from Human Preferences," ACM Transactions on Human-Robot Interaction, 2024 (arXiv 2402.15757; https://arxiv.org/abs/2402.15757) uses DPP batch selection for pairwise comparisons. CONFIRMED metadata. Directly relevant to BT-style reward learning.

Support: well supported (T14 has strong core papers). Direct evidence that randomisation specifically helps at low budget is indirect: Kirsch notes added noise helps batches, and Hacohen/Gupte discuss low-budget behaviour. I found no paper isolating "randomisation helps at 10-60 labels".

## T15. Parametric head + GP/kernel; GP uncertainty on frozen features

1. Wilson, A. G., Hu, Z., Salakhutdinov, R., Xing, E. P. "Deep Kernel Learning." AISTATS 2016 (PMLR 51, pp. 370-378; arXiv 1511.02222). https://proceedings.mlr.press/v51/wilson16.html . CONFIRMED.
   Applies: a GP on top of learned/pretrained features. With frozen features this reduces to a GP with a kernel on embeddings, plus optional learned base-kernel hyperparameters. It supplies the GP-uncertainty route for acquisition.

2. Houlsby, N., Huszar, F., Ghahramani, Z., Lengyel, M. "Bayesian Active Learning for Classification and Preference Learning." arXiv 1112.5745, 2011 (preprint). https://arxiv.org/abs/1112.5745 . CONFIRMED metadata. No formal venue seen. Content on preference sections not read.
   Applies: introduces BALD and applies it to GP classification and GP preference learning. The closest classical match for GP-based active preference learning.

3. Chu, W. & Ghahramani, Z. "Preference Learning with Gaussian Processes." ICML 2005. https://dl.acm.org/doi/10.1145/1102351.1102369 ; PDF https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf . CONFIRMED.
   Applies: the GP counterpart of the BT model (GP prior on latent utility, pairwise likelihood, Laplace inference). It sketches information-gain active selection as future work. A kernel BT model on frozen embeddings is a natural ensemble partner for the parametric head.

4. Huseljic et al. "Efficient Bayesian Updates for Deep Active Learning via Laplace Approximations." Springer chapter (ECML PKDD 2025 per search summary); arXiv 2210.06112. https://arxiv.org/abs/2210.06112 . Venue and author list UNVERIFIED (first author per secondary summary only).
   Applies: last-layer Laplace (equivalent to a GP with a linear kernel on frozen features) with closed-form updates on new labels, a cheap look-ahead mechanism for T13 as well.

Also seen: Gupte et al. 2024 (above) for the linear-head-on-frozen-features AL baseline.

Support: thin as a direct match. I found NO paper that benchmarks an ensemble of a parametric head and a GP, or a GP on CLIP/DINOv2 embeddings, for small-data active learning. Only the building blocks above exist. Treat any claim that "this combination is validated" as unsupported.

## T16. Ties / "no preference" labels, calibration vs ranking; temperature scaling for BT

1. Liu, J., Ge, D., Zhu, R. "Reward Learning From Preference With Ties." arXiv 2410.05328, Oct 2024 (preprint). https://arxiv.org/html/2410.05328 ; https://www.arxiv.org/pdf/2410.05328 . CONFIRMED metadata; content from search summary.
   Applies: Bradley-Terry-with-ties (BTT). Argues that omitting ties distorts preference-strength estimates (reward differences shrink or are mismeasured). That supports a calibration, not ranking, effect of tie labels.

2. Rao, P. V. & Kupper, L. L. "Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model." JASA 62(317), 1967. https://www.tandfonline.com/doi/abs/10.1080/01621459.1967.10482901 . And Davidson, R. R. "On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments." JASA 65(329), 1970. https://www.tandfonline.com/doi/abs/10.1080/01621459.1970.10481082 . CONFIRMED metadata.
   Applies: the two standard tie likelihoods (threshold parameter vs draw-probability). Use them to turn "no preference" labels into a proper likelihood term.

3. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. "On Calibration of Modern Neural Networks." ICML 2017 (PMLR 70, pp. 1321-1330). https://arxiv.org/abs/1706.04599 . CONFIRMED.
   Applies: temperature scaling (one scalar fit on held-out NLL, preserves ordering, so AUC/accuracy unchanged). For a BT head, scale the reward difference by 1/T. With 10-60 labels, fit T by cross-validation or the training fold.

4. "Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions", arXiv 2607.13423 (2026). https://arxiv.org/abs/2607.13423 . UNVERIFIED: seen only in a search listing and a secondary summary; authors unknown; fetch failed. It is about classification, not reward models. Reports that temperature scaling fit on hard labels does worse against soft human-label distributions.
   Applies if confirmed: soft or tie labels as calibration targets.

Support: thin for the specific claim. I found NO paper showing that tie or "no preference" labels improve log-loss but not AUC in pairwise models. The mechanism (ties add mass near 0.5, which affects probabilities but barely changes ordering) is a plausible hypothesis and is consistent with the temperature-scaling argument and the Liu et al. result, but it should be tested empirically by the project and not cited as established. Ties/BT-extension and temperature-scaling background themselves are well supported. A general ranking-calibration pattern (pointwise logistic losses good for ECE/LogLoss, worse for ranking) appears in "Regression Compatible Listwise Objectives for Calibrated Ranking," arXiv 2211.01494 (https://arxiv.org/pdf/2211.01494); authors/venue UNVERIFIED.

## Summary of support
- T13: well supported for method papers (Roy & McCallum, NTK look-ahead, Freytag EMOC, Settles EGL); thin for preference/ranking applications (Donmez & Carbonell only).
- T14: well supported (Stochastic Batch Acquisition, BatchBALD, Hacohen low-budget, Gupte foundation-model AL, Biyik batch preference AL). Direct low-budget randomisation evidence is only indirect.
- T15: thin for the exact combination; building blocks (DKL, GP preference learning, BALD, last-layer Laplace) are solid.
- T16: background (BT ties, temperature scaling) well supported; the specific "ties help log-loss, not AUC" claim has no direct source found.

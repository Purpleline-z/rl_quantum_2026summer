# Literature for active BT reward learning on RHEED (frozen SimCLR features, MLP head, 4 type heads, ties / not_apply)

Verification note: WebFetch could not resolve hosts (DNS ENOTFOUND for arxiv.org, aclanthology.org, mlr.press) in this sandbox, so no page was opened directly. Every paper below was seen in WebSearch results, with the URL, title and author/venue data shown there. Details in "how it helps" lines that come from secondary summaries are flagged. Items I could not confirm are marked UNVERIFIED.

## T1. Multi-attribute / multi-objective / coupled BT; active selection for multi-attribute

1. **Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (ArmoRM)** - Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, Tong Zhang. Findings of EMNLP 2024. https://arxiv.org/abs/2406.12845 (also https://aclanthology.org/2024.findings-emnlp.620/).
   Helps: a frozen backbone with a linear multi-objective head is the same architecture as our "frozen features + per-type heads". Caveat: it regresses on absolute ratings and uses a prompt-conditioned gate, neither of which we have.
2. **Bradley-Terry and Multi-Objective Reward Modeling Are Complementary** - Zhang et al. (first-author name taken from the ML Anthology key "zhang2026iclr"; not otherwise confirmed). ICLR 2026. https://arxiv.org/abs/2507.07375 (https://openreview.net/forum?id=3QHKJcwnpb).
   Helps: one shared backbone with a BT head plus a multi-attribute head trained jointly; the paper argues the two losses help each other when data is scarce. This is the closest published example of coupling heads through shared parameters. It is evidence for sharing structure across our 4 type-heads, not a hierarchical BT prior.
3. **Comparison-based Active Preference Learning for Multi-dimensional Personalization (AMPLe)** - Minhyeon Oh, Seungjoon Lee, Jungseul Ok. ACL 2025 (Long), pp. 33145-33166. https://aclanthology.org/2025.acl-long.1590/ ; arXiv https://arxiv.org/abs/2411.00524 ; code https://github.com/ml-postech/AMPLe.
   Helps: the only paper found that does active comparison selection over several reward dimensions (each dimension is an attribute). The query rule (halving the preference space with a Bayesian update) is for a linear-in-features reward over a few dimensions, so it fits a Bayesian last-layer head. The query-selection details come from a search summary; the paper was not read.
4. **Active Reward Modeling / Reviving The Classics** (see T2 item 4). Single-attribute, but its D-optimal criterion decomposes naturally per head and over a shared last layer.
5. Peripheral: Defresne et al., "Preference Elicitation for Multi-objective Combinatorial Optimization with Active Learning and Maximum Likelihood Estimation", IJCAI 2025, https://arxiv.org/abs/2503.11435. It uses a BT-MLE model with ensemble-based pair acquisition. Different setting (combinatorial optimization), so only a loose analogy.

Support for T1: moderate on the shared-head architecture, thin on active selection. I found no paper on hierarchical or coupled BT with a shared prior across attribute heads, and none on active selection of comparisons for multi-attribute reward models trained with BT and ties. AMPLe is the only close match.

## T2. Bayesian last-layer / Laplace, empirical Bayes, prior-centred regularisation, Fisher-based active reward modeling

1. **Laplace Redux - Effortless Bayesian Deep Learning** - Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig. NeurIPS 2021. https://arxiv.org/abs/2106.14806
   Helps: recommends post-hoc last-layer Laplace (KFAC/GGN) as a cheap default, and gives the `laplace` PyTorch library. This is directly usable on the MLP head.
2. **Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks** - Agustinus Kristiadi, Matthias Hein, Philipp Hennig. ICML 2020. https://arxiv.org/abs/2002.10118 (https://proceedings.mlr.press/v119/kristiadi20a.html)
   Helps: justifies a Gaussian posterior over only the last layer (here the small BT head). It does not cover the preference likelihood.
3. **Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning** - Alexander Immer, Matthias Bauer, Vincent Fortuin, Gunnar Raetsch, Mohammad Emtiyaz Khan. ICML 2021 (PMLR 139, 4563-4573). https://arxiv.org/abs/2104.04975 ; code https://github.com/aleximmer/marglik
   Helps: gives evidence-maximisation (empirical Bayes) for the prior precision (per layer) using a Laplace and Gauss-Newton marginal likelihood. This is the method to choose the prior precision without a validation split, which matters with only ~168 groups.
4. **Active Reward Modeling: Adaptive Preference Labeling for Large Language Model Alignment** - Yunyi Shen, Hao Sun, Jean-Francois Ton (author names from my memory; the search results confirmed the title and venue but did not list authors, so treat the names as UNVERIFIED). ICML 2025, PMLR 267. https://proceedings.mlr.press/v267/shen25c.html ; OpenReview https://openreview.net/forum?id=GSyX4amBFR ; arXiv preprint "Reviving The Classics: Active Reward Modeling in Large Language Model Alignment" https://arxiv.org/abs/2502.04354
   Helps: this is the closest match to our setting. It applies Fisher-information D-optimal design to the last linear layer of a BT reward model on fixed embeddings, and reports D-opt beating entropy, max-difference, coreset and BatchBALD. Note that the Fisher information of a BT model is p(1-p)(phi_A - phi_B)(phi_A - phi_B)^T, so it favours pairs that are both uncertain and spread out in feature space. Caveat: a 2026 follow-up (https://arxiv.org/abs/2602.01581, "Nearly Optimal Active Preference Learning and Its Application to LLM Alignment", seen only as a search result, not read) argues that D-optimal design is non-adaptive.
5. **Explicit Inductive Bias for Transfer Learning with Convolutional Networks (L2-SP)** - Xuhong Li, Yves Grandvalet, Franck Davoine. ICML 2018. https://arxiv.org/abs/1802.01483 (https://proceedings.mlr.press/v80/li18a.html)
   Helps: penalise ||w - w0||^2 toward the pretrained solution rather than toward 0. For our head this is the same as a Gaussian prior centred on a reference solution (a previous round's or a pooled-across-types solution) with the precision set by item 3. Reported gains are largest at low data (from the search summary; check the paper).
6. **Bayesian Reward Models for LLM Alignment** - Adam X. Yang et al. (venue: ICML 2024 workshop per ML Anthology). https://arxiv.org/abs/2402.13210
   Helps: Laplace approximation over a reward model's LoRA weights, used for uncertainty and for reducing over-optimisation. Evidence that Laplace on a reward model is workable; it does not do active selection.
7. **Uncertainty Estimation for Language Reward Models** - Adam Gleave, Geoffrey Irving. arXiv 2022. https://arxiv.org/abs/2203.07472
   Helps as a caution: ensemble-based active learning did not beat random sampling for reward models, and the uncertainty only weakly predicted error. A Laplace/Fisher criterion must be checked against a random-sampling baseline.
8. Optional: **Bayesian Low-rank Adaptation for Large Language Models** (Yang et al., ICLR 2024), https://arxiv.org/abs/2308.13111. It says fine-tuned models are overconfident on small datasets and Laplace helps, which supports our small-data motivation.

Support for T2: strong. I found no paper that combines last-layer Laplace, evidence-maximised prior precision and Fisher-based acquisition in one reward model. Combining items 1-5 is a fair claim of novelty, but it should be worded as "we combine established components".

## T3. Ties and abstention

1. **Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model** - P. V. Rao, L. L. Kupper. JASA 62(317), 194-204, 1967. https://www.tandfonline.com/doi/abs/10.1080/01621459.1967.10482901
2. **On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments** - Roger R. Davidson. JASA 65(329), 317-328, 1970. https://www.tandfonline.com/doi/abs/10.1080/01621459.1970.10481082
   Helps (1 and 2): the two standard 3-outcome extensions; Rao-Kupper has a threshold parameter, Davidson has a geometric-mean tie term. Both use one extra scalar per head (type), which is cheap with ~500 judgments.
3. **Reward Learning From Preference With Ties** - Jinsong Liu, Dongdong Ge, Ruihao Zhu. arXiv 2024 (an aggregator lists ICLR 2025; not confirmed). https://arxiv.org/abs/2410.05328
   Helps: a BT-with-ties (Rao-Kupper) reward model; shows that dropping ties can bias the measured preference strength. Caveat: tie labels were LLM-simulated on HH-RLHF, so there is no real expert tie evidence.
4. **On Extending Direct Preference Optimization to Accommodate Ties** - Jinghong Chen, Guangyu Yang, Weizhe Lin, Jingbiao Mei, Chenxu Lyu, Bill Byrne (author list differs by source; first and last are consistent). NeurIPS 2025 (arXiv 2409.17431). https://arxiv.org/abs/2409.17431 ; https://openreview.net/forum?id=h71cSd2loX
   Helps: replaces BT with Rao-Kupper and with Davidson in DPO; reports that keeping tied pairs instead of discarding them avoids the degradation seen in plain DPO (translation and summarisation). Closest evidence that tie data helps.
5. **A Statistical Framework for Ranking LLM-Based Chatbots** - Siavash Ameli, Siyuan Zhuang, Ion Stoica, Michael W. Mahoney. ICLR 2025. https://arxiv.org/abs/2412.18407
   Helps: about 20% of Chatbot Arena comparisons are ties; models ties with Rao-Kupper/Davidson generalised to pair-specific tie parameters. Notes that the usual half-win handling is ad hoc. A reviewer summary says the main gains are in-sample, so held-out benefit is not settled.
6. Context: Davidson-Luce model for multi-item choice with ties, https://arxiv.org/pdf/1909.07123 (seen in search results; authors not confirmed).

Support for T3: strong for the classical models, moderate for ML use. I found nothing that handles a "not_apply" / abstain outcome, which differs from a tie: "the attribute does not apply to this pair" is missing-at-random-style censoring, not a 0.5 outcome. The standard choice is to drop not_apply pairs from that head's likelihood and keep their other heads' labels. No cited paper does this. There is no real-human-tie evidence that ties improve held-out reward models; the evidence is synthetic (item 3) or on DPO (item 4).

## T4. GP preference learning, BALD, deep features, small data

1. **Preference Learning with Gaussian Processes** - Wei Chu, Zoubin Ghahramani. ICML 2005, pp. 137-144. https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf ; https://dl.acm.org/doi/abs/10.1145/1102351.1102369
   Helps: the probit-likelihood GP preference model with Laplace inference and evidence-based hyperparameter selection. A GP over frozen SimCLR features is the non-parametric analogue of our MLP head. The Laplace inference detail comes from a search summary of a later tutorial.
2. **Bayesian Active Learning for Classification and Preference Learning** - Neil Houlsby, Ferenc Huszar, Zoubin Ghahramani, Mate Lengyel. arXiv 2011 (1112.5745). https://arxiv.org/abs/1112.5745
   Helps: introduces BALD and extends it to GP preference learning by recasting a pair as a classification problem. This is the acquisition rule to try on the Laplace head (mutual information between the pair outcome and the weights).
3. **Deep Bayesian Active Learning for Preference Modeling in Large Language Models (BAL-PM)** - Luckeciano C. Melo, Panagiotis Tigas, Alessandro Abate, Yarin Gal. NeurIPS 2024. https://arxiv.org/abs/2406.10023
   Helps: BALD on top of frozen-LLM features with a small Bayesian head, plus a diversity term on the feature space; reports 33-68% fewer labels and that naive BALD picks redundant samples. Very close to our setup; the diversity term suggests batching with a feature-space repulsion.
4. **Preferential Bayesian Optimization** - Javier Gonzalez et al. (the search showed only the first author; the other names are not confirmed). ICML 2017. https://arxiv.org/abs/1704.03651
   Helps: GP with a Bernoulli (duel) likelihood and acquisition functions for pairs. This is the standard way to apply a GP preference model to an experimental problem, but it targets optimisation, not held-out accuracy.
5. **Active Preference-Based Gaussian Process Regression for Reward Learning** - https://arxiv.org/abs/2005.02575 (seen in search results; authors not confirmed; robotics).
6. Peripheral: Takeno et al., preferential BO with skew GPs, ICML 2023, https://github.com/CyberAgentAILab/preferentialBO (repository page only; the exact posterior is a skew GP and plain Laplace may be poor). Lun Chau et al., "Learning Inconsistent Preferences with Gaussian Processes", AISTATS 2022, https://proceedings.mlr.press/v151/lun-chau22a.html (intransitive preferences). Active Preference Learning for LLMs (Muldrew et al., ICML 2024), https://arxiv.org/abs/2402.08114 (entropy-based acquisition on DPO).

Support for T4: strong for the foundations (items 1, 2, 4), moderate for modern use (item 3). I found no paper that puts a GP preference model on frozen deep features for small-data expert preference learning, and none on GP preference learning with ties. Our setting looks open here too.

## Summary of gaps
- Hierarchical or coupled BT across attribute heads with a shared prior: not found beyond shared-backbone multi-head work (T1 items 1-2).
- Active selection for multi-attribute BT reward models with ties and not_apply: not found; AMPLe is closest.
- Treatment of an "attribute does not apply" outcome: no literature found.
- Real human tie data improving held-out reward models: no direct evidence.
- Gleave & Irving and BAL-PM both warn that naive uncertainty sampling can fail to beat random, so include a random baseline and a diversity term.

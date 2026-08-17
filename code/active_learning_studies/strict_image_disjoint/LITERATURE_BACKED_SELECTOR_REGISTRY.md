# Approved Selector Registry

This benchmark uses no custom acquisition method. Candidate preference labels are unavailable to every selector.

| Local selector | Published method | Local scoring rule |
|---|---|---|
| `random` | Standard random baseline | Uniform sampling without replacement. |
| `predictive_entropy` | Uncertainty sampling | Mean Bradley--Terry Bernoulli entropy across reconstruction heads. |
| `core_set_k_center` | Core-Set, Sener and Savarese (2018) | Greedy farthest-first k-center in pair-embedding space. |
| `mc_dropout_mutual_information` | BALD, Gal et al. (2017) | MC-dropout mutual information of pairwise probabilities. |
| `mc_dropout_probability_variance` | MC-dropout uncertainty, Gal and Ghahramani (2016) | Variance of pairwise probabilities across stochastic dropout passes. |

Excluded implementations are not part of this benchmark: uncertainty-diversity, cluster-quota uncertainty, reward variance, metadata and mixture scores, utility prediction, and Cluster-Margin adaptation.

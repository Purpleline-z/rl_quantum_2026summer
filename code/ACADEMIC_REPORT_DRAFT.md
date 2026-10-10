# Active Selection from Pairwise Image Preferences

### Abstract

This study examines how to select a limited number of image-pair preference labels for training a Bradley--Terry reward model of RHEED reconstruction types, and what evidence can separate acquisition strategies. The reward model has one head per reconstruction type; four types are active (1×1, c(6×2), (√13×√13), HTR) and Twinned(2×1) is excluded from ideal-image partitions and evaluation. An SHA-256 content-identity audit enforces that no outer-test image appears in training pairs, candidate images, reference anchors, utility validation, or negative anchors. Three results frame the study. (1) With the encoder fine-tuned end to end (§5.10), eight acquisition strategies on five seeds cannot be separated: accuracy on 28 ideal test images is 0.4–0.7, training noise (±0.1, including dependence on the order of the selected pairs) exceeds the differences between strategies, and at the largest budget all strategies train on identical data. (2) With a frozen encoder and an order-independent head (§5.11), type accuracy on ideal images reaches about 0.85, but the reference anchors alone give 0.848 with no pair labels at all and pair labels add nothing; this endpoint cannot compare acquisition rules. (3) On a held-out preference endpoint that pair labels do move (§5.12; 33 strategy variants, 35 seeds, two acquisition conditions), information-based rules built on the last-layer posterior (Laplace BALD, BALD × P(decisive)) give a small, significant gain over random selection when one batch is chosen (log-loss about 0.04 lower, AUC about 0.01 higher) that does not reproduce under sequential retraining and, under a second training schedule, persists only as a smaller AUC gain for Laplace BALD; uncertainty-driven rules do not beat random choice, and graph-cut selection is worse. Repeating the comparison with the 20% pair-level hold-out of the earlier classifier2 work (§5.13) gives a different set of significant rules (only core-set), so which coverage- or information-based rule helps depends on the split. (4) Those comparisons counted the budget in pair groups and revealed all of a selected group's judgments (about 3.1), whereas the labelling software asks for one pair and one type per query; repeating the comparison with one (pair, type) judgment as the query unit and the budget in judgments (§5.14) again finds no rule significantly better than random on log-loss, calibrated log-loss or AUC in either split except two in one table (an all-head Cluster-quota rule on log-loss and deep-ensemble BALD × P(decisive) on AUC, Split B sequential), and the significant rules of §5.12–5.13 do not reproduce. A pre-registered replication on 35 new seeds confirms a modest 1–3 point gain in decisive-judgment accuracy for two rules that choose the type with the model (deep-ensemble BALD × P(decisive), Fisher D-optimal) and the AUC gain of the first in Split B sequential, but not the log-loss effects. (5) With a head schedule suited to 10–60 judgments, pool-wide variance-reduction (optimal-design) rules do beat random selection on ranking metrics (§5.15; with a gentler head, lr 0.0003, I-optimal design improved log-loss by 0.044, AUC by 0.021 and accuracy by 0.021, all p < 0.0001 on 35 fresh splits): in pre-registered runs on fresh splits (seeds 800–834) I-optimal design, Fisher D-optimal design and BALD × P(decisive) all improved decisive accuracy by about 0.015–0.017 and AUC by 0.007–0.013 (all Holm < 0.02), with log-loss gains of 0.00–0.035; the gains are 2–3 times larger from a cold start (10 random initial judgments), and a priority weight on HTR (soft type targeting) raised HTR accuracy by 0.067 and AUC by 0.058 without hurting the other types. These results are for resampled splits of the same 168 pair groups, not new data, and plain uncertainty sampling and coverage rules do not share the effect. (6) Graph-aware selection was tested last (§5.16–5.20). The graph of the comparisons itself is nearly a matching (168 pair groups, 284 images, 117 connected components), so a ranking network on it has nothing to recover. Selection rules on an image-similarity graph (ten nearest neighbours of the frozen SimCLR features, with the type of the typed reference images propagated over it to predict which judgments will be decisive) gave no significant gain over random selection that can be attributed to the graph under the old head schedule (§5.16–5.18; first-generation rules on 35 seeds; three rules frozen on development seeds and confirmed once on 25 unseen seeds with the pre-registered criterion not met; replications on 20 and 30 fresh seeds with a gain of about +0.03 to +0.04 log-loss in split B that was not significant after correction). Under the re-tuned head schedule of (5) the frozen type-aware rules do beat random selection (§5.20): the plain coverage rule in all four blocks on 30 fresh seeds (log-loss +0.027 to +0.050, Holm p ≤ 0.009), in the two split-B blocks on 35 further seeds and in two of four blocks (after correction over the blocks) on 40 more, and pooled over 105 fresh seeds in all four blocks (+0.014 to +0.037, Holm p < 0.01); the graph-derived type posterior adds a little over a type-only decisive predictor (AUC +0.009 to +0.013) but not distinguishably from the non-graph variance-reduction design of (5). A graph-regularised version of that design (Laplacian prior on the last layer) beats random selection under the gentler schedule on two independent seed sets (35 and 40 seeds; log-loss +0.031 to +0.044, Holm p < 0.002), but its shuffled-graph control and the non-graph design do the same, so the gain cannot be attributed to the graph; its advantage on development seeds did not replicate under the re-tuned schedule. A pre-registered fixed-epoch comparison (3 vs. 30 epochs, lr=10^{-4}) is reported for five seeds (42, 79, 123, 202, 303).

## 1. Introduction

Pairwise labels ask a simple question: *given two images, which one is preferred for a specified reconstruction type?* They are often easier for experts to provide than absolute scores. However, expert time is limited, so an active-learning system must choose which hidden comparisons to reveal.

We study the following computational question: among **candidate pairs**, which pairs should be labelled and added to a preference-trained image model? 

Figure 1 shows the intended evaluation firewall. Validation can select training settings and a stopping epoch; the outer test is used only after these choices are frozen.

![Pipeline and evaluation protocol](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/pipeline_and_evaluation_protocol.svg)

*Figure 1. Computational pipeline and evaluation firewall.*

## 2. Method

### 2.1 Preference reward model

See https://github.com/ymeng3/Quantum/tree/main/Classifier2.

Each image $x$ is passed through a ResNet-18 encoder and a reward head that returns one score per reconstruction class, $r_\theta(x,t)$, with five output dimensions corresponding to TYPE\_ORDER = [(1×1), Twinned(2×1), c(6×2), (√13×√13), HTR]. For a labelled pair $(x_i,x_j)$ of type $t$, the Bradley--Terry model assigns the probability that the first image is preferred as

$$
P_\theta(x_i \succ x_j \mid t) = \sigma\!\left(r_\theta(x_i,t)-r_\theta(x_j,t)\right),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

In plain language, the model is confident that the first image should win when its learned score is much larger than the second image's score. For a first-image win, the loss is

$$
\mathcal L_{BT} = -\frac{1}{|D|}\sum_{(i,j,t)\in D} w_{ij}\log P_\theta(x_i \succ x_j\mid t),
$$

where $w_{ij}$ is an optional annotator-confidence weight. The implementation also handles reversed wins, ties, and not-applicable labels using the corresponding loss terms in `pairwise_active_learning_pipeline.py`.

#### Downstream reconstruction prediction and accuracy

The downstream endpoint is reconstruction-type accuracy on held-out ideal images, not pairwise training accuracy. For a test or utility-validation image $x$, let $R_c$ be the set of ideal reference anchors labelled as class $c$, and let $R_{-c}=\bigcup_{d\ne c}R_d$ be the anchors from every *other* class. The class-$c$ win rate is

$$
W_c(x)=\frac{1}{|R_{-c}|}\sum_{z\in R_{-c}}\sigma\!\left(r_\theta(x,c)-r_\theta(z,c)\right).
$$

$r_\theta(x,c)$ is the $c$-th reward-head score for image $x$. Each term asks whether $x$ is more compatible with class $c$ than an anchor known to belong to another class, on the class-$c$ reward dimension. $W_c(x)$ averages that comparison over all non-$c$ reference anchors. The predicted reconstruction type is

$$
\hat c(x)=\underset{c\in\mathcal C}{\arg\max}\ W_c(x),
$$

where $\mathcal C$ is the available reconstruction-type set. Thus a class wins only when its reward dimension separates the image from reference examples of the competing classes; the model does not compare an image with same-class reference anchors at this step. For an evaluation split $E$ with absolute labels $c(x)$, downstream accuracy is

$$
A(E)=\frac{1}{|E|}\sum_{x\in E}\mathbf{1}\!\left[\hat c(x)=c(x)\right].
$$

In plain language, this is the fraction of held-out ideal images assigned the correct reconstruction type. $A(E)$ is calculated identically for `utility_validation` and `outer_test`; only the former is available while selecting epochs, learning rates, and acquisition settings. The implementation is [Experiment.evaluate](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pairwise_active_learning_pipeline.py#L447-L464), which also saves correct count, total count, and per-class accuracy.

The downstream model is a ResNet-18 encoder followed by a 512-to-256-to-5 reward head (Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→5)), fine-tuned end to end with AdamW (weight decay $10^{-4}$, batch size 16): all encoder and head parameters are updated, and the encoder is **not frozen** during training. Frozen (cached) encoder embeddings are used only by the acquisition rules that need distances between candidate images (core-set, cluster-based rules); they never enter training. At prediction time, the reward heads score an image against the labelled reference images (win-rate rule above). The five reward heads correspond to all five entries in TYPE\_ORDER. Twinned(2×1) is excluded from ideal image splits and evaluation; its reward head (index 1) is present in the model and receives gradient updates from any Twinned-labelled pairwise rows in the training pool. For seeds 42, 79, and 123, the pipeline at git SHA 58d59d6 did not filter Twinned pairwise rows from the training pool or candidate pool; those rows trained the Twinned head alongside the four active classes. All strategies were affected identically, so the relative comparison between strategies is valid. Seeds 202 and 303 run under commit 482f712, which adds a Twinned pairwise filter to load\_and\_split(). Evaluation metrics use only the four active class heads in all cases.

The two encoder initializations are different starting representations under this same training procedure. The shipped SimCLR checkpoint is image-only self-supervised pretraining and has not seen pairwise preference labels. ImageNet initialization uses torchvision ResNet-18 weights trained with ImageNet-1K supervision. Pairwise labels enter only during reward-model fine-tuning.

The fixed-epoch comparison (Section 5.9) pre-registers lr=$10^{-4}$, weight decay=$10^{-4}$, and epoch counts $E\in\{3,30\}$. The validation-selected protocol (Task 3b, Section 5.8) uses the utility-validation grid to select lr and epoch count per encoder and budget; the outer test is never consulted during that selection.

### 2.2 Active selection

Candidate comparisons retain their images but hide their human winner labels. Active selection decides which $b$ comparisons should be labelled next so that the retrained preference model makes more accurate reconstruction-type predictions on held-out ideal images. Let $L$ be the currently labelled pair groups, let $C$ be the label-hidden candidate groups, let $S_b\subseteq C$ be the $b$ groups selected by an acquisition strategy, and let $Y(S_b)$ be the labels revealed only after selection. The quantity that selection is trying to increase is

$$
\Delta A_{\mathrm{val}}(S_b\mid L)=
A_{\mathrm{val}}\!\left(F(L\cup Y(S_b))\right)
-A_{\mathrm{val}}\!\left(F(L)\right).
$$

Here $F(L)$ is the model produced by the fixed training procedure using labelled groups $L$, $A_{\mathrm{val}}$ is validation reconstruction accuracy, and $\Delta A_{\mathrm{val}}$ is the improvement attributable to the newly labelled batch. An acquisition rule cannot calculate this quantity directly because $Y(S_b)$ is hidden; instead, it uses the available images, embeddings, and model predictions as a proxy. Section 2.3 defines those proxies. Validation data choose a training schedule and compare design choices; the locked outer test is evaluated only after those choices are fixed.

The current study is a single-round, batch active-learning experiment: train on $L$, score all of $C$, select one batch $S_b$, reveal its labels, retrain on $L\cup Y(S_b)$, and evaluate. This makes a strategy's contribution easy to compare at a fixed annotation budget.

Implementation entry point: [Experiment.select](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pairwise_active_learning_pipeline.py#L536).

~~~text
Algorithm 3: Single-round active-selection loop
Input: labelled groups L, label-hidden candidate groups C, budget b,
       acquisition rule A, validation set V, locked outer-test set T
1. Train a reward model f on L using a validation-selected schedule.
2. Compute permitted label-free quantities for every c in C.
3. Select S_b = A(C, f, L, b); do not read winner labels in this step.
4. Reveal Y(S_b), then retrain f' on L union Y(S_b).
5. Measure A_val(f') on V for protocol decisions and save the full run record.
6. Once all strategies and settings are frozen, evaluate the chosen protocol on T.
Output: selected groups S_b, validation result, and one locked-test result.
~~~

### 2.3 Acquisition strategies and their roles

All methods select complete pair groups, never individual images. The candidate view contains image paths and permitted model outputs, but not the human winner label.

| Strategy | Selection rule | Why it may help | Main tuning or failure mode |
|---|---|---|---|
| Random | Uniform sample of candidate groups | Essential unbiased baseline | High variance at small budgets |
| Uncertainty | Largest mean predictive entropy | Requests comparisons near the model decision boundary | Can repeatedly select visually similar pairs |
| Core-set | Farthest-first coverage in embedding space | Covers underrepresented image regions | Depends on encoder geometry and distance metric |
| Cluster-quota uncertainty | Spread selections over clusters, then rank by uncertainty | Avoids spending all labels in one region | Cluster count and quota can be mismatched to the data |
| Uncertainty + diversity | Largest $a(c)$ | Balances difficult and nonredundant pairs | Requires validation choice of $\lambda$ |
| Cluster-Margin | Low margin, then round-robin from small clusters | Keeps ambiguous pairs while protecting rare clusters | Historical evidence is a separate extension, not pooled with the five-strategy curve |
| MC-dropout variance / mutual information | Rank disagreement across dropout predictions | Models epistemic uncertainty | Requires dropout probability and Monte-Carlo sample-count calibration |

The common strategy dispatcher is [`Experiment.select`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pairwise_active_learning_pipeline.py#L536). Each pseudocode block gives the exact implemented score or sampling distribution, defines its symbols, and links to the implementing function. $C$ is the candidate-pair pool, $L$ is the labelled-pair pool, and $b$ is the requested number of selected pair groups throughout.

#### Algorithm 3a: Random baseline

Implementation: [`random_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L110).

```text
Input: candidate pair groups C, budget b, random seed s
initialize a deterministic random-number generator with s
return b distinct groups sampled uniformly from C
```

$$
\Pr(S)=\binom{|C|}{b}^{-1}\quad\text{for every }S\subseteq C\text{ with }|S|=b.
$$

Thus every possible batch $S$ of $b$ groups has the same probability; $|C|$ is the number of available candidates. Random has no model-derived score and provides the reference for whether a more elaborate rule earns its complexity.

#### Algorithm 3b: Predictive uncertainty

Implementation: [`uncertainty_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L116), which calls [`score_uncertainty`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L75).

```text
Input: candidates C, trained reward model f, budget b
for each pair (xi, xj) in C:
    logits_h = r_h(xi) - r_h(xj)   for h = 1,...,5
    p_h = sigmoid(logits_h)
    u_h = BernoulliEntropy(p_h)
    uncertainty = mean(u_1,...,u_5)
return the b pairs with largest uncertainty
```

$$p_h(c)=\sigma\!\left(r_\theta(x_i,h)-r_\theta(x_j,h)\right),\qquad u_h(c)=-p_h(c)\log p_h(c)-(1-p_h(c))\log(1-p_h(c)),\qquad u(c)=\frac{1}{H}\sum_{h=1}^{H}u_h(c).$$

For candidate $c=(x_i,x_j)$, $p_h(c)$ is the predicted probability that $x_i$ wins under reward head $h$, $H=5$ is the number of reward heads, and $u(c)$ is the mean Bernoulli entropy over all heads. The label-free candidate view (candidate\_metadata) does not carry a reconstruction-type index; the mean-head formula is therefore used in every production run. Large entropy means the current model finds the comparison hard on average across all reward dimensions.

#### Algorithm 3c: Core-set coverage

Implementation: [`core_set_select`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_shared_calculations.py#L52); pair embeddings are created through [`pair_vector`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_shared_calculations.py#L20).

```text
Input: candidate pair embeddings C, labelled-pair embeddings L, budget b
selected = empty
repeat b times:
    choose c in C that has the greatest distance to L union selected
    add c to selected
return selected
```

$$
v(c)=\frac{e(x_i)+e(x_j)}{2},\qquad
c^*=\underset{c\in C\setminus S}{\arg\max}\ \min_{z\in\{v(\ell):\ell\in L\}\cup\{v(s):s\in S\}}\|v(c)-z\|_2.
$$

Here $e(x)$ is the encoder embedding of image $x$, $v(c)$ is the average embedding of the two images in pair $c$, $S$ is the batch selected so far, and $\|\cdot\|_2$ is Euclidean distance. The rule repeatedly adds the candidate furthest from already covered labelled or selected pairs. When $L$ is empty, the implementation initializes distances from the candidate-pool mean.

#### Algorithm 3d: Cluster-quota uncertainty

Implementation: [`cluster_quota_uncertainty_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L122).

```text
Input: candidates C with cluster IDs, reward model f, budget b
compute uncertainty for every candidate
allocate a proportional integer quota, with a minimum of one before budget is exhausted
within each cluster quota, choose the most uncertain remaining candidate
fill any unused budget with globally most uncertain remaining candidates
return selected groups
```

$$
n_g=|\{c\in C:g(c)=g\}|,\qquad
q_g=\max\!\left(1,\mathrm{round}\!\left(b\frac{n_g}{|C|}\right)\right).
$$

$g(c)$ is the cluster assigned to pair $c$, $n_g$ is that cluster's candidate count, and $q_g$ is its proportional provisional quota. Within each cluster the implementation takes the $q_g$ largest uncertainty scores $u(c)$, then fills any remaining budget with the largest $u(c)$ globally. The minimum-one rule gives small clusters a chance to contribute before the budget is exhausted.

#### Algorithm 3e: Uncertainty plus diversity

Implementation: [`uncertainty_diversity_select`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_shared_calculations.py#L29).

```text
Input: uncertainty-scored candidates C, labelled embeddings L, budget b, lambda
for each candidate c:
    u = normalized predictive uncertainty of c
    d = normalized distance from c to L and already selected pairs
    score(c) = u + lambda*d
repeat until b groups are selected:
    add remaining candidate with largest score and update diversity distances
return selected groups
```

$$
d(c;L,S)=\min_{z\in\{v(\ell):\ell\in L\}\cup\{v(s):s\in S\}}\|v(c)-z\|_2,
\qquad a(c)=R(u(c))+\lambda R(d(c;L,S)).
$$

Here $u(c)$ is the uncertainty from Algorithm 3b, $d(c;L,S)$ is distance from covered pair embeddings, $S$ is updated after every selection, and $\lambda$ weights diversity. $R(\cdot)$ is the implementation's robust normalization: values are clipped at their 5th and 95th percentiles and rescaled to $[0,1]$. The rule greedily chooses the remaining candidate with the largest $a(c)$ and then recomputes diversity distances.

#### Algorithm 3f: Cluster-Margin

Implementation: [`cluster_margin_pairwise_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L145).

```text
Input: candidates C with clusters, reward model f, budget b
compute each pair's distance from probability 0.5 (its margin)
prefilter min(10*b, number of candidates) pairs with the smallest margins
round-robin across prefiltered clusters, visiting smaller prefiltered clusters first
return b selected groups
```

$$
m(c)=\frac{1}{H}\sum_{h=1}^{H}\left|p_h(c)-\tfrac{1}{2}\right|,
$$

$p_h(c)$ and $H$ have the meanings defined for predictive uncertainty; $H=5$ reward heads; the mean is over all five. $m(c)$ is small when the heads place the comparison close to a 50--50 decision. Let $P$ contain the $\min(10b,|C|)$ candidates with the smallest margins. The implementation groups $P$ by $g(c)$, orders the groups from smallest to largest, and takes their smallest-margin remaining member in round-robin order until $b$ pairs are selected.

#### Algorithm 3g: MC-dropout probability variance

Implementation: [`score_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L33) and [`select_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L77).

```text
Input: candidates C, reward model f with dropout active, b, M stochastic passes
for m = 1,...,M:
    score every candidate with a different dropout mask
for each candidate:
    compute variance of its predicted preference probability across M passes
return b candidates with largest probability variance
```

$$
\bar p_h(c)=\frac{1}{M}\sum_{m=1}^{M}p_{mh}(c),\qquad
v_{\mathrm{MC}}(c)=\frac{1}{H}\sum_{h=1}^{H}\frac{1}{M}\sum_{m=1}^{M}\left[p_{mh}(c)-\bar p_h(c)\right]^2.
$$

$p_{mh}(c)$ is the preference probability for candidate $c$ from dropout pass $m$ and reward head $h$, $M$ is the number of stochastic dropout passes, and $H$ is the number of heads. A large $v_{\mathrm{MC}}(c)$ means predictions change substantially when dropout perturbs the model, so the candidate is selected as epistemically uncertain.

#### Algorithm 3h: MC-dropout mutual information

Implementation: the same [`score_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L33) and [`select_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L77) functions, dispatched with metric `mc_dropout_mutual_information`.

```text
Input: candidates C, reward model f with dropout active, b, M stochastic passes
obtain M stochastic preference distributions per candidate
for each candidate:
    predictive_entropy = entropy(mean probability)
    expected_entropy = mean entropy(stochastic probabilities)
    mutual_information = predictive_entropy - expected_entropy
return b candidates with largest mutual_information
```

$$
I_{\mathrm{MC}}(c)=\frac{1}{H}\sum_{h=1}^{H}\left[
h\!\left(\frac{1}{M}\sum_{m=1}^{M}p_{mh}(c)\right)
-\frac{1}{M}\sum_{m=1}^{M}h\!\left(p_{mh}(c)\right)\right],
\qquad h(p)=-p\log p-(1-p)\log(1-p).
$$

The symbols $p_{mh}(c)$, $M$, and $H$ are as above. $h(p)$ is Bernoulli entropy. The first term measures uncertainty after averaging dropout predictions; the second measures their average individual uncertainty. Their difference is large when model samples disagree, which directs labels toward uncertainty caused by model parameters rather than one consistently ambiguous pair.

### 2.4 Leakage-safe training algorithms

```text
Algorithm 1: Build leakage-safe partitions
Input: ideal images, pairwise rows, seed
1. Compute SHA-256 identity for every image file.
2. Assign each unique ideal identity to exactly one of {reference, validation, outer test}.
3. Collect identities assigned to outer test.
4. Remove every pairwise/unlabelled-trajectory row and negative anchor whose image identity is in outer test.
5. Form pair-disjoint initial and candidate pair groups; image reuse across those pair groups is allowed.
6. Audit and save every path/hash overlap count; fail a run if any test-identity overlap remains.
Output: pair-disjoint training/candidate groups and identity-disjoint ideal partitions.
```

```text
Algorithm 2: Train for a fixed number of epochs
Input: labelled pairs, reference images, utility-validation set, epoch count E
for epoch = 1,...,E:
    optimize Bradley--Terry and permitted reference-anchor losses for one epoch
    record mean training loss
    metric = reconstruction_accuracy(utility-validation)
    record metric; never evaluate outer test here
return model after E epochs
```


## 3. Data Protocol and Leakage Prevention

### 3.1 Historical data and benchmark contract

The v1.8 preference source contains 669 valid rows representing 179 unordered pair groups. The legacy controlled benchmark used 50 initial pair groups and 120 candidate groups. The current identity-safe Task 3 study uses 10 initial pair groups and up to 100 candidate pair groups (from simclr\_three\_seed\_identity\_safe\_task3\_settings.json); the remaining groups are unused. There is no separate pairwise validation partition: every preference group is initial, candidate, or unused. The [Stage 1 manifest](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/study_manifest.json) records the exact allocation.

The unit of acquisition is an unordered pair group; initial and candidate groups are pair-disjoint. The same image may occur in different non-test pair groups because image-disjoint pair partitions are unnecessarily restrictive for this application. Ideal reference, utility-validation, and outer-test images are separately partitioned. In contrast, no outer-test identity may appear in training pairs, candidate images, reference anchors, utility validation, or Bad-image anchors.

For the default identity-safe configuration, `Twinned(2 x 1)` is excluded and 144 unique identities remain from 150 ideal-image files. The resulting capacity is 28 outer-test images, 28 utility-validation images, and 88 reference anchors. Counts by reconstruction type are exported in the [identity-safe split-capacity CSV](active_learning_studies/pair_disjoint_not_image_disjoint/results/identity_safe_protocol/ideal_split_capacity.csv).

| Reconstruction type | Raw files | Unique identities | Outer test | Utility validation | Reference anchors |
|---|---:|---:|---:|---:|---:|
| `(1 x 1)` | 41 | 41 | 8 | 8 | 25 |
| `c(6 x 2)` | 42 | 41 | 8 | 8 | 25 |
| `(√13 x √13)` | 38 | 36 | 7 | 7 | 22 |
| `HTR` | 29 | 26 | 5 | 5 | 16 |
| **Total** | **150** | **144** | **28** | **28** | **88** |

### 3.2 Pairwise CSV cleaning: removal of trajectory images identical to ideal images

During the identity audit for seeds 202 and 303, the SHA-256 audit reported non-zero values for two fields that must be zero: `pairwise_image_identity_overlap_reference` (3) and `pairwise_image_identity_overlap_utility_validation` (2). These fields count the number of pairwise training rows in which at least one trajectory image is content-identical (same SHA-256 hash) to an image assigned to the reference-anchor or utility-validation partition.

The root cause was that 8 trajectory image files in `Trajectories/` had identical byte content to ideal image files in `STO_ideal_*/` folders. These 8 trajectory images appeared in 40 rows of the v1.8 pairwise CSV (`Quantum Label Data - Pairwise_Comparisonv1.8.csv`). If any of those rows entered the training or candidate pool, the model would be trained on preference comparisons containing images that are byte-for-byte identical to held-out evaluation images—a form of label-free test-set leakage.

Seeds 42, 79, and 123 did not trigger this violation because, for those random seeds, the affected ideal images were assigned to the outer-test partition, where the pipeline's `pairwise_image_identity_overlap_outer_test` filter removes overlapping pairs from the training pool automatically. Seeds 202 and 303 assigned some of those same ideal images to the reference and utility-validation partitions instead, where no analogous runtime filter was active, causing the audit to fail.

The fix was applied directly to the source CSV rather than adding a runtime workaround: the 40 affected rows were deleted from `Quantum Label Data - Pairwise_Comparisonv1.8.csv`, reducing the row count from 678 to 638. The 8 trajectory images involved span the `(1 x 1)` and `c(6 x 2)` reconstruction classes. After this removal, all seven forbidden-overlap audit fields are zero for seeds 202, 303, and all previously passing seeds. The cleaned CSV is committed at git SHA `5ca1bb4` and all subsequent runs use it.

This cleaning has no effect on seeds 42, 79, and 123 results reported in §5.8–5.10: those runs predate the cleaning but the removed pairs were already excluded from their training pools by the outer-test filter, so no training set changes for those seeds.

### 3.3 Identity-safe split contract

![Leakage-safe split protocol](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/leakage_safe_split_protocol.svg)

*Figure 2. Required split protocol. Identity means SHA-256 file content, not filename.*

The implementation enforces this rule in `active_learning_program/pairwise_active_learning_pipeline.py`. It retains audit counts for path and content-identity overlaps. This avoids the failure mode in which duplicated image bytes appear under different paths and silently leak from a train-time pool into the test set.

## 4. Implementation and Reproducibility

The reusable program is in `active_learning_program/`. `pairwise_active_learning_pipeline.py` loads preference CSVs, creates ideal partitions, trains the reward model, and executes selectors. `resumable_model_training.py` stores per-epoch training loss and utility-validation accuracy. The pipeline trains for a fixed number of epochs (selected per encoder and budget by the Task 3b validation grid) with no early stopping. The budget-aware runner uses `utility_validation` for training decisions and records `outer_test_not_evaluated: true` during calibration.

Each run should save: configuration; data hashes; pair manifests; split audit; per-epoch metrics; selected pair IDs; seed; data freeze manifest (generate\_data\_freeze\_manifest.py); and all aggregate CSVs used in figures. The freeze manifest records git commit hash, source-file SHA-256s, reconstruction classes included and excluded, image-identity partition assignments, pair-group assignments, partition counts by class, and all seven forbidden-overlap audit fields. It fails loudly if any overlap constraint is violated or if Twinned(2×1) appears in any partition. Every paper number must trace to one of these saved artifacts.

## 5. Results and Evidence Status

### 5.1 Historical selector curves: diagnostic only

The repository contains five-seed, fixed-schedule curves across acquisition budgets. Figure 3 is generated directly from `results/selected_summaries/budget_curve_5seed/tables/strategy_performance_by_acquisition_budget.csv`.

![Historical budget curve](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/historical_selector_budget_curve.png)

*Figure 3. Five-strategy accuracy across acquisition budgets from the fixed-schedule study; the winner changes with budget.*

**Question.** Does the same acquisition rule select useful labels at every annotation budget? This historical five-seed curve uses the unmodified-image mode and the fixed full-model schedule described in Section 2.1. Mean outer-test accuracy identifies the leading arm at each budget:

| Acquired pair groups | Leading arm | Mean outer-test accuracy | Decision carried into the identity-safe rerun |
|---:|---|---:|---|
| 10 | Random | 0.413 | Retain random as the paired baseline when labels are scarce. |
| 25 | Uncertainty | 0.440 | Compare against random under the validation-selected schedule. |
| 50 | Cluster-quota uncertainty | 0.460 | Retain this documented coverage comparator. |
| 75 | Uncertainty + diversity | 0.513 | Preserve the exact diversity weight as a validation-tuned setting. |
| 100 | Uncertainty | 0.647 | Use uncertainty as a principal reference arm. |

The full means and standard deviations are in the [strategy-by-budget table](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv).

The changing winners indicate that annotation budget changes the selection problem. At 10 labels, random sampling provides a stable cross-section of the pool; at 50, cluster quotas can prevent the batch from collapsing into one embedding region; at 75--100, uncertainty-based rules can sample several ambiguous regions. Test these explanations with paired seed-level gains over random after validation selects the training schedule, alongside a fixed-total-update control that removes the effect of differing optimizer updates.

### 5.2 Symmetry preprocessing is an experimental factor

The repository evaluates three input modes: `none`, `left_half_mirror`, and `symmetric_average`. The latter two encode a symmetry assumption by reconstructing an image from its left half or averaging it with its horizontal reflection. They can remove discriminative asymmetry and reduce nuisance variation.

![Historical symmetry factorial](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/historical_symmetry_factorial.png)

*Figure 4. Strategy--budget--symmetry interaction from the Stage 2 aggregate CSV; no one preprocessing mode wins across conditions.*

**Question.** Does enforcing horizontal symmetry make candidate comparisons easier to use? The answer depends on both budget and selector. The table below gives the strongest and weakest condition at each budget from the [Stage 2 factorial](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage2_symmetry_factorial/aggregate/strategy_budget_summary.csv).

| Budget | Highest mean accuracy condition | Lowest mean accuracy condition | Operational meaning |
|---:|---|---|---|
| 10 | core-set + symmetric average: 0.533 | core-set + none: 0.307 | Averaging can make embedding coverage more stable when labels are scarce. |
| 25 | random + symmetric average: 0.507 | core-set + none: 0.347 | The transform can help even without model-based selection, consistent with removing condition-specific variation. |
| 50 | core-set + left-half mirror: 0.527 | uncertainty + none: 0.300 | Mirroring changes which regions look distinct in embedding space. |
| 75 | uncertainty-diversity + none: 0.513 | random + none: 0.293 | The unmodified image retains useful information for the best selector at this budget. |
| 100 | uncertainty + none: 0.647 | core-set + symmetric average: 0.367 | Strong symmetry processing can remove distinctions that uncertainty exploits. |

`left_half_mirror` and `symmetric_average` change the information given to the encoder. A gain after averaging indicates that suppressing asymmetry helps that selector and budget; a loss indicates that the removed asymmetric detail helps the comparison. Test this interaction with an identity-safe strategy × budget × preprocessing factorial using the same validation-selected schedule for all arms at a given budget.

### 5.3 SimCLR versus ImageNet initialization

![Historical encoder-screen validation accuracy](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/historical_encoder_screen.png)

*Figure 5. Utility-validation encoder screen; error bars are standard deviations over three seeds.*

**Question.** Which frozen starting representation gives the selector a more useful coordinate system? In the three-seed validation screen, ImageNet initialization exceeds SimCLR for both random selection (0.556 vs. 0.389) and uncertainty selection (0.611 vs. 0.300), as shown in the [encoder-screen CSV](active_learning_studies/pair_disjoint_not_image_disjoint/results/protocol_diagnostics/encoder_initialization_screen/aggregate/encoder_utility_validation_summary.csv).

The encoder determines which pairs appear close, which candidate clusters exist, and which comparisons look uncertain. In this screen, ImageNet initialization produced higher utility-validation accuracy after fine-tuning. Compare encoder × strategy with identity-safe seeds, optimizer, stopping rule, and budget held fixed; report per-seed paired differences to distinguish a broad shift from a few favourable splits.

### 5.4 PCA and t-SNE representation diagnostics

**Question.** Do frozen image features put examples of the same reconstruction type in the same local neighborhood, and do the labelled ideal images cover the trajectory population? An embedding is a numerical location assigned to each image: local label consistency means an image's nearest neighbours usually have the same reconstruction label. PCA displays the two directions with the most variation; the retained-variance fraction tells the reader how much of the full representation is visible in that two-axis sketch. t-SNE emphasizes who is near whom and is useful for local neighborhoods, but its apparent gap widths are not literal distances.

![PCA of frozen SimCLR features](active_learning_studies/image_representation_analysis/results/representation_exploration/figures/rheed_simclr_resnet18_pca.png)

*Figure 6. PCA of frozen RHEED-SimCLR features for the exploratory image manifest.*

![t-SNE of frozen SimCLR features](active_learning_studies/image_representation_analysis/results/representation_exploration/figures/rheed_simclr_resnet18_tsne.png)

*Figure 7. t-SNE of frozen RHEED-SimCLR features for the same exploratory manifest.*

![PCA of frozen ImageNet features](active_learning_studies/image_representation_analysis/results/representation_exploration/figures/imagenet_resnet18_pca.png)

*Figure 8. PCA of frozen ImageNet ResNet-18 features for the exploratory image manifest.*

![t-SNE of frozen ImageNet features](active_learning_studies/image_representation_analysis/results/representation_exploration/figures/imagenet_resnet18_tsne.png)

*Figure 9. t-SNE of frozen ImageNet ResNet-18 features for the same exploratory manifest.*

The full frozen-feature check gives 5-NN accuracy 0.884 for SimCLR and 0.896 for ImageNet, versus 0.832 for raw pixels. The more specific coordinate diagnostics now show what that average hides. In PCA coordinates, `(1 x 1)` has the highest 5-NN recall for both SimCLR (0.927) and ImageNet (0.976); `c(6 x 2)` is also locally consistent (0.857 and 0.881). These are the classes whose nearest feature-space neighbours usually share their label, so coverage- or similarity-based acquisition has a meaningful local geometry to work with for them.

The main ambiguous boundary is `HTR` versus `RT13`: SimCLR PCA 5-NN misclassifies 7 HTR images as RT13 and 9 RT13 images as HTR; ImageNet PCA makes the same pair of errors 5 and 7 times. Their nearest-other-class distances are also small relative to within-class spread in the [class-separation table](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/class_separation_metrics.csv), placing many HTR and RT13 images in mixed frozen-feature neighborhoods. Use a boundary-pair acquisition slice to test whether expert preference labels resolve HTR--RT13 comparisons, alongside an image-only versus permitted-additional-feature ablation.

`Twinned(2 x 1)` has only four labelled ideal images and PCA recall of 0.000 for SimCLR and 0.250 for ImageNet. Four examples provide too few same-class neighbours for a stable five-neighbour neighborhood. Obtain or annotate additional Twinned ideal images before comparing encoder separation for this type.

![Per-class PCA-coordinate 5-NN recall](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/per_class_knn_recall_pca.png)

*Figure 10. Which ideal-image types have locally label-consistent PCA neighborhoods? `(1 x 1)` and `c(6 x 2)` are high-recall in both encoders; the small Twinned sample is not. Generated from [per-class 5-NN diagnostics](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/per_class_knn_metrics.csv).*

![ImageNet PCA-coordinate 5-NN confusion](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/imagenet_resnet18_pca_knn_confusion.png)

*Figure 11. Which reconstruction types share local ImageNet-feature neighborhoods? The off-diagonal HTR--RT13 counts identify the principal boundary for targeted comparisons. Generated from the [confusion matrix](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/knn_confusion_matrix.csv).*

For trajectory coverage, the near threshold is the 95th percentile of each labelled ideal image's leave-one-out nearest-ideal distance, rather than an arbitrary radius. In the PCA views, 71.7% of trajectory frames lie inside the SimCLR labelled-ideal neighborhood and 81.9% lie inside the ImageNet neighborhood; the remaining 28.3% and 18.1% are feature-space regions sparsely represented by the ideal set. These frames are candidates for a trajectory coverage audit or expert labeling before treating an ideal-image classifier as representative of the whole trajectory stream. The full thresholds and fractions are in [trajectory-neighborhood coverage](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/trajectory_neighborhood_coverage.csv).

![Trajectory neighborhood coverage](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/trajectory_neighborhood_coverage.png)

*Figure 12. Fraction of trajectory frames near the labelled ideal-image neighbourhood in each two-dimensional view. The plot identifies coverage gaps to inspect; it does not assign labels to unlabeled trajectory frames.*

PCA retains 76.2% of the SimCLR feature variation but only 37.4% of ImageNet variation, so the ImageNet PCA plot is a more compressed sketch of its full representation. Use the two-dimensional plots to locate candidate overlap and coverage questions, then test those questions with acquisition and validation experiments. All metrics, coordinate files, and thresholds are recorded in the [diagnostic manifest](active_learning_studies/image_representation_analysis/results/representation_exploration/section5_diagnostics/manifest.json).

### 5.5 Historical utility, redundancy, and batch interaction

**Question.** Why can a pair or a batch look useful in an intermediate utility analysis but not yield the highest final accuracy? The historical individual-pair utility used in the exploratory runs is

$$
U(p\mid L)=A_{\mathrm{outer}}\!\left(F(L\cup\{p\})\right)-A_{\mathrm{outer}}\!\left(F(L)\right).
$$

Here $p$ is one candidate pair, $L$ is the current labelled set, $F$ is the fixed retraining procedure, and $A_{\mathrm{outer}}$ is accuracy on the historical ideal-image outer-test set. This quantity asks what happens when one pair is added by itself. The realised utility of a selected batch $B$ is $U(B\mid L)=A_{\mathrm{outer}}(F(L\cup B))-A_{\mathrm{outer}}(F(L))$. It is not the sum of the individual utilities because pairs are retrained together; their gradients and the later selection sequence can change one another's effect. These outer-test utilities are historical diagnostics only and are not used by the current selector or budget-aware calibration.

The earlier budget-30 observation comes from one sequential trace, where each round changes $L$ before the final endpoint is measured. Its intermediate single-pair utilities and later final accuracy therefore condition on different labelled sets. The [sequential log](active_learning_studies/pair_disjoint_not_image_disjoint/results/earlier_explorations/sequential_exploration_20260728/epoch%202%20budget%2030%20batch%202%20seed%2042%20experiment_log.csv) identifies a question for the controlled study; it does not separate redundancy, interaction, or overfitting.

![Historical 15-seed utility and accuracy comparison](active_learning_studies/pair_disjoint_not_image_disjoint/results/strategy_followup_analysis/fifteen_seed_extension/fifteen_seed_strategy_comparison.png)

*Figure 13. Historical 15-seed budget-100 comparison. The figure asks whether the strategy that improves batch utility also improves post-acquisition accuracy; uncertainty has the largest mean on both measures.*

At budget 100, uncertainty has the largest historical mean post-acquisition accuracy (0.522) and batch utility (+0.144) over 15 seeds. Uncertainty--diversity follows at 0.489 and +0.111; random at 0.460 and +0.082; cluster-diverse at 0.427 and +0.049; and core-set at 0.422 and +0.044, as reported in the [15-seed summary](active_learning_studies/pair_disjoint_not_image_disjoint/results/strategy_followup_analysis/fifteen_seed_extension/fifteen_seed_summary.csv). Post-acquisition-accuracy standard deviations range from 0.121 to 0.188, and the per-seed outcomes therefore vary across runs.

![Historical pair similarity versus batch utility](active_learning_studies/pair_disjoint_not_image_disjoint/results/selected_summaries/diversity_and_coverage/figures/pair_similarity_vs_batch_utility.png)

*Figure 14. Historical pair-similarity versus batch-utility diagnostic. The figure asks whether less similar selected pairs explain utility; the strategy means do not show a monotonic relationship.*

Redundancy does not explain the observed ranking by itself. Core-set has the lowest mean image-reuse rate (0.082) and uncertainty--diversity lowers it relative to random (0.091 versus 0.094), yet their mean batch utilities (+0.044 and +0.111) remain below uncertainty (+0.144). Mean pair cosine similarities are tightly grouped from 0.947 to 0.949 despite the utility spread, in the [redundancy table](active_learning_studies/pair_disjoint_not_image_disjoint/results/strategy_followup_analysis/redundancy_analysis/redundancy_utility_accuracy.csv). Thus “more diverse” is not sufficient evidence that a batch provides more reconstruction-relevant supervision.

![Historical PCA of selected core-set pairs](active_learning_studies/pair_disjoint_not_image_disjoint/results/selected_summaries/diversity_and_coverage/figures/core_set_selected_pairs_pca.png)

*Figure 15. Historical PCA of selected pair embeddings. Overlap between core-set and uncertainty selections in this two-dimensional view does not test the full-dimensional farthest-first distances used by core-set.*

The PCA overlap between core-set and uncertainty selections does not conflict with the core-set algorithm: core-set chooses farthest-first pairs using full pair embeddings, whereas the plot compresses those embeddings to two coordinates. The actual concern is different: distant regions in the frozen embedding may not correspond to comparisons that improve reconstruction accuracy. Resolve that question with an identity-safe experiment that, for each selected batch, records individual utilities, the realised batch utility, and

$$
I(B\mid L)=U(B\mid L)-\sum_{p\in B}U(p\mid L).
$$

$I(B\mid L)$ measures non-additive batch interaction. Evaluate whether it varies with image reuse, pair similarity, cluster coverage, and training/validation curves. A negative interaction term supports harmful combination effects; a validation decline while pairwise training accuracy rises supports overfitting; neither explanation can be assigned from the saved historical batch records because their individual utilities are unavailable.

### 5.6 Metadata fusion: a separate, currently data-limited study

**Question.** Do process-monitor variables contain prediction information absent from the image?

Synchronized process-monitor variables (substrate temperature, deposition rate, chamber pressure, and gas-flow setpoints) are collected during growth but are not yet available in structured form for the image sets used in Tasks 3b and 3c. Metadata fusion is therefore deferred: this study uses image-only reward model inputs throughout. Once structured metadata is available and linked to trajectory frames, the question can be addressed by augmenting the reward model's encoder input with a process-variable embedding and comparing image-only versus image+metadata utility-validation accuracy at each acquisition budget.

### 5.7 Physics-informed soft constraints for a complete trajectory

**Question.** Can known ordering preferences improve the interpretation of a *sequence* of classifier outputs without changing the image model? The trajectory study takes calibrated probabilities for the five active states $(1\ x\ 1)$, `Bad`, $c(6\ x\ 2)$, $(\sqrt{13}\ x\ \sqrt{13})$, and `HTR` (Twinned(2×1) excluded, consistent with the reward model's evaluation scope), then chooses the highest-scoring complete path. Its [configuration](active_learning_studies/rheed_trajectory_ordering_analysis/higher_order_trajectory_constraint_configuration.json) specifies three soft preferences: the first state is `(1 x 1)`; before a `Bad` frame occurs, a path should not move from `(1 x 1)` directly to another reconstruction; and a path should not leave `HTR` once it has entered it.

For an ordered trajectory of $T$ frames, with classifier probability $q_t(s)$ for state $s$ at frame $t$, the decoder selects

$$
\hat y_{1:T} = \underset{y_{1:T}\in S^T}{\arg\max}\ \left[\sum_{t=1}^{T}\log q_t(y_t)-\phi_{\mathrm{start}}(y_1)-\sum_{t=2}^{T}\phi_{\mathrm{transition}}(y_{t-1},y_t,h_{t-1})\right].
$$

Here $S$ is the six-state set, $y_{1:T}$ is one possible state sequence, and $h_{t-1}$ records whether `Bad` has occurred earlier in that path. $\phi_{\mathrm{start}}$ penalizes a first state other than `(1 x 1)`. $\phi_{\mathrm{transition}}$ penalizes a non-`Bad`, non-`(1 x 1)` state before `Bad`, and a transition from `HTR` to any other state. A penalty subtracts evidence but never forbids a path: sufficiently stronger image probabilities can still select an exception. The weak, moderate, and strong settings multiply every base log penalty by $0.223$, $0.693$, and $1.609$, respectively; these correspond to evidence factors of $1.25$, $2$, and $5$.

```text
Input: ordered frame probabilities q[1:T, state], penalty level
For each possible first state s:
    score[1, s, seen_bad=(s == Bad)] = log q[1, s] - start_penalty(s)
For t = 2,...,T:
    For each previous state and Bad-history flag:
        For each current state:
            candidate = previous_score + log q[t, current]
            candidate -= penalty_if_leaving_HTR(previous, current)
            candidate -= penalty_if_skipping_Bad(history, current)
            retain the best candidate and its predecessor
Backtrack the best final state to obtain the decoded path
Write raw per-frame argmax, all decoded paths, changed-frame flags, and rule counts
```

The exact decoder is [decode_rheed_trajectory_with_higher_order_constraints.py](active_learning_program/decode_rheed_trajectory_with_higher_order_constraints.py); it uses dynamic programming with a one-bit `Bad has occurred` memory. This separation is deliberate in implementation terms: decoder outputs do not alter pairwise labels, model weights, metadata features, or acquisition scores.

The completed evidence is a [filename/order audit](active_learning_studies/rheed_trajectory_ordering_analysis/results/temporal_constraint_audit/temporal_audit_report.md), which records frame-order parsing, missing indices, duplicate indices, and ambiguity. It does not establish the frequency of any physical transition. A fully quantitative decoded-trajectory evaluation requires an externally validated five-state classifier with a calibrated `Bad` probability saved for ordered frames; that evaluation is deferred until the reward model trained in Task 3c is transferred to the trajectory stream. The decoder implementation is complete and ready for that integration.

### 5.8 Task 3b: budget-aware training protocol

**Question.** Should every acquisition budget use the same learning rate and number of epochs? Task 3a answers this with a validation-only grid: 3 seeds (42, 79, 123) × 5 budgets × 4 learning rates × 3 epoch counts = 180 cells, SimCLR initialization only. Each cell starts from ten labelled pair groups, adds a deterministic random batch of pair groups up to its budget, trains, and measures utility-validation accuracy; the outer test is never opened. Task 3b then picks, for each budget, the setting with the highest mean validation accuracy. The complete grid is in the [validation-calibration summary](active_learning_studies/pair_disjoint_not_image_disjoint/results/simclr_three_seed_identity_safe_task3/frozen_task3b_protocol/three_seed_validation_calibration_summary.csv); the selection is saved as the [Task 3b schedule](active_learning_studies/pair_disjoint_not_image_disjoint/results/simclr_three_seed_identity_safe_task3/frozen_task3b_protocol/simclr_three_seed_budget_specific_schedule.json).

| Budget | Selected epochs | Selected lr |
|---:|---:|---:|
| 10 | 30 | $3\cdot10^{-4}$ |
| 25 | 10 | $3\cdot10^{-4}$ |
| 50 | 30 | $3\cdot10^{-4}$ |
| 75 | 10 | $3\cdot10^{-4}$ |
| 100 | 10 | $3\cdot10^{-4}$ |

Ties were broken by fewer epochs, then lower learning rate. The schedule was selected on seeds 42, 79, and 123 only, so seeds 202 and 303 are out-of-sample for it. In Task 3c, every strategy at a given budget uses the same selected setting, so strategies differ only in which pair groups they acquire.

### 5.9 Fixed-epoch single-shot comparison (pre-registered)

**Question.** Does training for 30 epochs instead of 3 consistently improve acquisition strategy performance across budgets?

The fixed-epoch comparison pre-registers lr=$10^{-4}$, weight decay=$10^{-4}$, and epoch counts $E\in\{3,30\}$ without consulting outer-test results. Two strategies are compared at each budget: random (the paired baseline) and uncertainty. Both use the same 10-pair initial pool, the same 100-pair candidate pool, and the same SHA-256-enforced identity-safe splits as Task 3c. The outer test (28 images, 4 classes) is evaluated after all protocol choices are frozen.

Mean outer-test accuracy over all five seeds (42, 79, 123, 202, 303; outer-test total = 28 per seed):

| Strategy | Budget | 3-epoch accuracy | 30-epoch accuracy | Δ (30 − 3) |
|---|---:|---:|---:|---:|
| Random | 10 | 0.379 ± 0.190 | 0.329 ± 0.073 | −0.050 |
| Random | 25 | 0.214 ± 0.067 | 0.564 ± 0.166 | +0.350 |
| Random | 50 | 0.236 ± 0.070 | 0.571 ± 0.197 | +0.336 |
| Random | 75 | 0.379 ± 0.096 | 0.557 ± 0.234 | +0.179 |
| Random | 100 | 0.457 ± 0.188 | 0.671 ± 0.130 | +0.214 |
| Uncertainty | 10 | 0.279 ± 0.077 | 0.364 ± 0.152 | +0.086 |
| Uncertainty | 25 | 0.336 ± 0.120 | 0.500 ± 0.025 | +0.164 |
| Uncertainty | 50 | 0.214 ± 0.107 | 0.557 ± 0.249 | +0.343 |
| Uncertainty | 75 | 0.357 ± 0.094 | 0.357 ± 0.101 | +0.000 |
| Uncertainty | 100 | 0.329 ± 0.030 | 0.621 ± 0.117 | +0.293 |

Over five seeds, 30 epochs consistently outperforms 3 epochs at budgets ≥ 25 for both strategies. At budget 10, the advantage is inconsistent (Δ = −0.050 for random, +0.086 for uncertainty), reflecting high variance with a small labelled set. The uncertainty advantage from the three-seed analysis (all Δ > 0) does not fully replicate: at budget 75, the five-seed mean is identical (0.357) for both epoch counts. The full per-seed results are in [pre\_registered\_outer\_test\_results\_at\_epochs\_3\_and\_30.csv](active\_learning\_studies/pair\_disjoint\_not\_image\_disjoint/results/simclr\_three\_seed\_identity\_safe\_task3/fixed\_epoch\_3\_and\_30\_single\_shot\_aggregate/pre\_registered\_outer\_test\_results\_at\_epochs\_3\_and\_30.csv) and the aggregate summary is in [fixed\_epoch\_outer\_test\_summary.csv](active\_learning\_studies/pair\_disjoint\_not\_image\_disjoint/results/simclr\_three\_seed\_identity\_safe\_task3/fixed\_epoch\_3\_and\_30\_single\_shot\_aggregate/fixed\_epoch\_outer\_test\_summary.csv).

### 5.10 Task 3c: eight-strategy acquisition comparison

**Question.** Starting from ten labelled pair groups, which of eight acquisition strategies gives the most accurate reconstruction-type classifier after acquiring 10, 25, 50, 75, or 100 more pair groups?

**Setup.** Each cell trains the reward model (§2.1) on the initial groups plus the groups one strategy selected, using the Task 3b schedule for that budget, and reports accuracy on the 28-image outer test (4 classes). There are 8 strategies × 5 budgets × 5 seeds (42, 79, 123, 202, 303) = 200 cells. Frozen selector parameters: cluster count 20 (10 for Cluster-Margin), diversity λ = 0.5, MC-dropout probability 0.2, 10 MC samples. All strategies share the same identity-safe splits. Seeds 202 and 303 ran under commit 482f712 (Twinned pairwise filter active); seeds 42, 79, and 123 ran at git SHA 58d59d6 (§2.1).

**How to read the results.** Three properties of this design limit what the table can show.

1. *Budget 100 does not compare strategies.* The candidate pool has 100 pair groups, so at budget 100 every strategy acquires the same 100 groups (one distinct selected set per seed). The eight budget-100 numbers per seed are repeated measurements of one data set; they differ only through training-run variation and span 0.39–0.68 (seed 42), 0.54–0.89 (seed 79), 0.46–0.75 (seed 123), 0.50–0.82 (seed 202), and 0.25–0.82 (seed 303). The budget-100 column is therefore a noise estimate, not a ranking.
2. *Training outcome depends on the order of the selected pairs.* Training is deterministic for a fixed ordered list (two runs on the same ordered list gave identical accuracy and per-epoch validation), but reordering the same pair groups changes the minibatches and therefore the model (outer-test 0.357 versus 0.429 on one 30-group example). Selected pairs are currently not sorted before training, so part of every strategy-to-strategy difference is order noise. This is a known limitation of the completed runs.
3. *The outer test is small.* With 28 images, one image is 0.036 accuracy; seed-level standard deviations in the table are 0.07–0.26.

**Results.** Mean outer-test accuracy ± standard deviation over five seeds (also saved as `task3c_outer_test_summary_by_strategy_and_budget.csv`):

| Strategy | Budget 10 | Budget 25 | Budget 50 | Budget 75 | Budget 100 |
|---|---:|---:|---:|---:|---:|
| Random | 0.507 ± 0.154 | 0.536 ± 0.175 | 0.536 ± 0.091 | 0.664 ± 0.096 | 0.679 ± 0.143 |
| Uncertainty | 0.421 ± 0.188 | 0.429 ± 0.080 | 0.579 ± 0.092 | 0.586 ± 0.117 | 0.621 ± 0.227 |
| Core-set | 0.486 ± 0.082 | 0.421 ± 0.069 | 0.564 ± 0.174 | 0.636 ± 0.102 | 0.679 ± 0.104 |
| Cluster-quota uncertainty | 0.479 ± 0.264 | 0.564 ± 0.127 | 0.636 ± 0.069 | 0.664 ± 0.109 | 0.564 ± 0.077 |
| Uncertainty + diversity | 0.536 ± 0.194 | 0.486 ± 0.128 | 0.650 ± 0.111 | 0.600 ± 0.159 | 0.721 ± 0.081 |
| Cluster-Margin | 0.421 ± 0.102 | 0.593 ± 0.155 | 0.486 ± 0.078 | 0.643 ± 0.143 | 0.671 ± 0.092 |
| MC-dropout variance | 0.386 ± 0.132 | 0.507 ± 0.191 | 0.650 ± 0.159 | 0.621 ± 0.178 | 0.514 ± 0.086 |
| MC-dropout mutual info | 0.393 ± 0.217 | 0.443 ± 0.206 | 0.693 ± 0.169 | 0.571 ± 0.156 | 0.700 ± 0.140 |

The highest mean at budgets 10, 25, 50, and 75 is Uncertainty + diversity (0.536), Cluster-Margin (0.593), MC-dropout mutual information (0.693), and a tie between Random and Cluster-quota uncertainty (0.664). At budgets 10 and 75, Random is within 0.03 of the leader, and at budget 75 it ties for first. The gaps between the leading strategies and Random are small compared with the standard deviations, and with the order noise described above, so these results do not establish that any acquisition rule outperforms random selection at this scale. Earlier three-seed (42, 79, 123) rankings, in which Cluster-quota uncertainty led at budget 10 and Core-set and MC-dropout variance at budget 75, did not hold once seeds 202 and 303 were added.

**Reference baseline.** A frozen SimCLR encoder with no training and no pairwise labels, classifying each outer-test image by its nearest reference image, reaches 0.871 ± 0.041 over the same five seeds' outer-test images ([results](active_learning_studies/pair_disjoint_not_image_disjoint/results/simclr_three_seed_identity_safe_task3/frozen_encoder_nearest_neighbour_baseline.csv), produced by `compute_frozen_encoder_nearest_neighbour_baseline.py`). This baseline uses absolute class labels of the reference ideal images, whereas the acquisition strategies choose pairwise preference labels, so they are not interchangeable. It does show that, in this protocol, the fine-tuned reward model (0.4–0.7) is well below what the frozen features alone support, and that the absolute accuracies in the table are limited by the training procedure, not by the encoder.

**Per-class accuracy at budget 100** (mean over five seeds; same data for all strategies, so differences are training noise):

| Strategy | (1×1) | c(6×2) | (√13×√13) | HTR |
|---|---:|---:|---:|---:|
| Random | 0.800 | 0.675 | 0.543 | 0.680 |
| Uncertainty | 0.700 | 0.750 | 0.343 | 0.680 |
| Core-set | 0.750 | 0.700 | 0.543 | 0.720 |
| Cluster-quota uncertainty | 0.500 | 0.700 | 0.400 | 0.680 |
| Uncertainty + diversity | 0.650 | 0.925 | 0.514 | 0.800 |
| Cluster-Margin | 0.750 | 0.700 | 0.514 | 0.720 |
| MC-dropout variance | 0.675 | 0.475 | 0.286 | 0.640 |
| MC-dropout mutual info | 0.900 | 0.750 | 0.343 | 0.800 |

(√13×√13) is the weakest class for every strategy (at most 0.543), consistent with the PCA diagnostic in §5.4, which shows it sharing a crowded neighbourhood with HTR in the frozen SimCLR feature space.

![Eight-strategy outer-test accuracy curves](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/eight_strategy_outer_test_accuracy_curve.png)

*Figure 16. Mean outer-test accuracy vs. annotation budget for all eight strategies (five seeds; bars are ± one standard deviation). At budget 100 all strategies use identical data, so the spread there is training noise.*

![Paired outer-test gain versus random](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/paired_outer_test_difference_vs_random.png)

*Figure 17. Paired seed-level accuracy difference relative to random for each strategy and budget. Positive values indicate the strategy outperforms random on the same seed; the distribution width reflects seed variance rather than strategy inconsistency.*

![Seed-level outer-test scatter](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/seed_level_outer_test_scatter.png)

*Figure 18. Outer-test accuracy for every individual seed × strategy × budget cell. The spread confirms that no single strategy dominates at every seed.*

All 200 cells are stored in `results/simclr_three_seed_identity_safe_task3/task3c_final_strategy_cells/`; Figures 16–18 are generated from them by `generate_simclr_identity_safe_task3_figures.py`. In the 80 cells of seeds 202 and 303, all seven forbidden-overlap audit fields are zero. The 120 cells of seeds 42, 79, and 123 record the other five fields as zero but do not record `pairwise_image_identity_overlap_reference` and `pairwise_image_identity_overlap_utility_validation`; §3.2 explains why those overlaps were already excluded for those seeds.

### 5.11 Why the encoder is frozen, and what it shows about type accuracy

**Question.** The fine-tuned reward model of §5.10 gave outer-test accuracies of 0.4–0.7 with noise as large as the differences between strategies. Is that noise a property of strategies, or of how the model is trained, and does a model trained differently change the answers?

**Why freeze.** Three observations from §5.10 motivate removing end-to-end fine-tuning from the strategy comparison. First, the same data can give different models: at budget 100 every strategy trains on the identical 100 pair groups, yet accuracy within a seed spans 0.25–0.89. Second, the result depends on the order of the selected pairs: 30 identical pair groups in two orders gave outer-test accuracies of 0.357 and 0.429 (training on a fixed ordered list is exactly repeatable; reordering the list changes the minibatches). Third, 34–342 labelled rows are very few for 11 million encoder parameters. Freezing the encoder and training only the reward head, with a loss that is a *sum* over rows, makes the trained head a function of the set of labelled pair groups rather than of their order.

**Method.** The SimCLR ResNet-18 encoder is applied once to every image and never updated; its 512-d features are cached (444 images across all seeds). The reward head is the same Linear(512→256)–ReLU–Dropout(0.2)–Linear(256→5) as before, but is trained by full-batch AdamW (weight decay $10^{-4}$) on the sum of per-row losses divided by the number of rows, with no dropout during training (dropout remains in the head so that MC-dropout acquisition can still sample it at selection time). The losses are those of §2.1: Bradley–Terry for decisive winners, an absolute-difference term for ties, a push-down term for `not_apply`, reference-anchor ranking (weight 0.25), and bad-image anchors (weight 0.10). A sum over rows is invariant to row order, which is verified by `test_frozen_encoder_order_invariance.py`. Acquisition code is unchanged: strategies read the cached features and the head. The candidate pool is enlarged from 100 to 158 groups (all groups not in the initial 10), so every budget up to 100 is smaller than the pool and strategies can genuinely differ. Learning rate $\in\{10^{-3},3\cdot10^{-3},10^{-2}\}$ and number of full-batch steps $\in\{100,300,1000\}$ were selected per budget on utility-validation accuracy with a deterministic random batch, on seeds 42, 79, and 123 only (outer test unopened), as in Task 3b:

| Budget | Selected lr | Selected steps | Mean validation accuracy |
|---:|---:|---:|---:|
| 10 | 0.01 | 300 | 0.857 |
| 25 | 0.003 | 300 | 0.881 |
| 50 | 0.01 | 100 | 0.905 |
| 75 | 0.003 | 100 | 0.893 |
| 100 | 0.001 | 300 | 0.869 |

**Residual noise.** With the data fixed, only the head's random initialisation varies. Across 8 initialisations the within-seed standard deviation of outer-test accuracy is 0.013–0.051 (mean 0.039), against 0.09–0.18 between strategies at budget 100 for the fine-tuned model; the between-seed standard deviation of the mean is 0.065 and is dominated by which 28 images fall in the outer test. The frozen model therefore removes most of the training noise but not the test-set sampling noise.

**Results: outer-test type accuracy.** Mean ± standard deviation over five seeds (same outer test, same identity-safe splits as §5.10; 200 cells; results in `results/frozen_encoder_task3/`):

| Strategy | Budget 10 | Budget 25 | Budget 50 | Budget 75 | Budget 100 |
|---|---:|---:|---:|---:|---:|
| Random | 0.836 ± 0.020 | 0.829 ± 0.016 | 0.779 ± 0.077 | 0.807 ± 0.065 | 0.821 ± 0.044 |
| Uncertainty | 0.821 ± 0.076 | 0.814 ± 0.047 | 0.843 ± 0.054 | 0.771 ± 0.054 | 0.807 ± 0.041 |
| Core-set | 0.829 ± 0.069 | 0.843 ± 0.065 | 0.779 ± 0.047 | 0.786 ± 0.067 | 0.821 ± 0.051 |
| Cluster-quota uncertainty | 0.807 ± 0.096 | 0.814 ± 0.105 | 0.757 ± 0.081 | 0.800 ± 0.070 | 0.807 ± 0.041 |
| Uncertainty + diversity | 0.864 ± 0.053 | 0.821 ± 0.044 | 0.764 ± 0.082 | 0.771 ± 0.090 | 0.829 ± 0.039 |
| Cluster-Margin | 0.843 ± 0.060 | 0.836 ± 0.054 | 0.786 ± 0.056 | 0.786 ± 0.098 | 0.793 ± 0.077 |
| MC-dropout variance | 0.829 ± 0.077 | 0.807 ± 0.070 | 0.829 ± 0.059 | 0.793 ± 0.059 | 0.836 ± 0.070 |
| MC-dropout mutual info | 0.807 ± 0.054 | 0.857 ± 0.044 | 0.729 ± 0.032 | 0.821 ± 0.036 | 0.829 ± 0.081 |

No strategy differs reliably from random. The paired differences (strategy minus random, same seed and budget) lie between -0.050 and +0.064; the largest in magnitude is uncertainty at budget 50 (+0.064, paired t = 2.45, n = 5), which is not significant once 35 comparisons are considered. Training on the initial 10 pair groups alone, with no acquisition, gives 0.871 ± 0.065, as high as any cell. Frozen-head accuracy (0.73–0.86) is also far above the fine-tuned model (0.4–0.7) and matches the frozen-encoder nearest-neighbour baseline of §5.10 (0.871 ± 0.041).

**Where the accuracy comes from.** The pattern above (more pair labels do not help, and the initial 10 groups already reach the plateau) suggests that type accuracy is determined by something other than the acquired pairs. Training the same frozen head with only some of its loss terms isolates the source (mean over five seeds, four head initialisations each):

| Training signal | Utility-validation accuracy | Outer-test accuracy |
|---|---:|---:|
| anchors only (0 pair groups) | 0.848 | 0.848 ± 0.038 |
| anchors + 10 groups | 0.864 | 0.852 ± 0.053 |
| anchors + 110 groups | 0.832 | 0.805 ± 0.076 |
| pairs only, 10 groups | 0.337 | 0.368 ± 0.114 |
| pairs only, 110 groups | 0.543 | 0.554 ± 0.110 |
| pairs only, all 168 groups | 0.389 | 0.409 ± 0.038 |

With **no pair groups at all**, the reference-anchor term alone reaches 0.848. Pairwise preferences alone reach only 0.37–0.55, and adding 110 groups to the anchors slightly lowers accuracy (0.805 versus 0.848). The anchors use the absolute class labels of about 88 reference ideal images per seed; the pairwise labels say which of two trajectory images better shows a reconstruction, which is information about quality, not about class identity. Type accuracy on ideal images is therefore a poor endpoint for judging which pairs to label: it is nearly saturated by the anchors, and the strategy comparisons of §5.10 and of this section cannot rank acquisition rules on it. §5.12 evaluates the quantity pair labels do teach.

### 5.12 Held-out preference prediction: comparing acquisition strategies on what pair labels teach

**Question.** Pair labels say which of two images better shows a reconstruction type. Which acquisition rules choose the pair groups that best predict *held-out human preferences*, once type accuracy on ideal images (saturated by the reference anchors, §5.11) is replaced by an endpoint the labels can move?

**Endpoint and split.** For every seed the 168 usable pair groups are divided into the 10 initial groups, 20 validation groups, 40 test groups, and a candidate pool of the remaining groups that share no image with the validation or test groups (64–77 groups, depending on the seed: 168 usable groups − 10 initial − 20 validation − 40 test = 98, from which 21–34 groups are removed because they share an image with a validation or test group; the largest budget, 60 groups, is therefore 78–94% of the pool). Validation and test groups are image-disjoint from the training and candidate groups and from each other, so no evaluation image is ever seen in training. The test endpoint is the reward model's prediction of the decisive judgments (winner 1 or 2; 49% of judgments are ties or not-applicable) in the test groups: for each decisive row, whether the head of that row's reconstruction type scores the winner higher. We report four numbers: (i) accuracy, (ii) Bradley–Terry log-loss, (iii) AUC of the signed score gap, which does not depend on the scale of the scores, and (iv) log-loss after fitting a single temperature on the validation groups, which separates ranking quality from score scale. Rows are weighted by their confidence weight.

**Training and calibration.** The reward model is the frozen-encoder head of §5.11. Learning rate and number of full-batch steps were chosen on validation log-loss with a random batch of the budget size, on seeds 42, 79, and 123 only, averaged over all four budgets; the test groups were not used. The selected setting is learning rate 0.01 and 100 steps.

**Strategies.** The eight strategies of §5.10 are used unchanged: Random, Uncertainty, Core-set, Cluster-quota uncertainty, Uncertainty + diversity, Cluster-Margin, MC-dropout probability variance, and MC-dropout mutual information. These original rules score a candidate with the head of the type of the group's first judgment row, which plays the role of the queried type (the annotator is asked to judge one reconstruction type per query). A group usually carries judgments for several types and all of them are acquired together with the group, so the budget counts groups, not judgments (about 3.1 judgments per group). Nineteen further strategies, in `new_pair_strategies.py`, aggregate over the four active heads, that is, they do not assume which type will be queried; a test confirms that they ignore the type. Seventeen are described below; the other two are deep-ensemble BALD (the mutual information between a preference and the head's random initialisation, from 8 heads trained on the same labels) and the same score weighted by the predicted probability of a decisive answer. They draw on the literature in `notes/literature_and_new_strategy_ideas.md`: relation-aware core-set (pair vector [(a+b)/2, |a−b|, a⊙b]), TypiClust, BADGE for a Bradley–Terry head, Fisher D-optimal design (Active Reward Modeling), Laplace-approximation BALD with batch updates, largest predicted gap and gap + posterior standard deviation (ActiveUltraFeedback-style), BALD weighted by a predicted probability that the judgment is decisive, DPP, ProbCover, MaxHerding, DropQuery, an image-coverage rule, an uncertainty-weighted facility-location rule on a kNN graph of pairs, FASS and graph cut. Six *all-head* versions of the original rules (suffix `_lf`: Uncertainty, Cluster-quota uncertainty, Uncertainty + diversity, Cluster-Margin, and the two MC-dropout rules) run the original code with the candidate's type removed and the unused Twinned head silenced (its last-layer row is zeroed so it adds the same constant to every candidate), which puts them on the same footing as the added strategies (neither assumes a queried type); 33 strategy variants in total are compared with Random, which is the mean of five independent draws per seed and budget so that its noise does not dominate the comparison.

**Two acquisition conditions.** *Single-shot*: one batch of the budget size is chosen using the model trained on the 10 initial groups, as in §5.10. *Sequential*: rounds of 10 groups, retraining the head on everything labelled so far before each round; the same trajectory is read out after 10, 20, 40, and 60 acquired groups. Sequential acquisition is how model-based rules are normally used and is cheap here because the encoder is frozen.

**Statistics.** The unit of replication is the seed (random split of ideal images, initial groups, validation and test groups). For each strategy and metric, the gain over Random is averaged over budgets within a seed, giving one number per seed; significance is a two-sided Wilcoxon signed-rank test on these numbers, with Holm correction across strategies. Five seeds (42, 79, 123, 202, 303) are the primary study seeds; 30 further seeds (400–429), outside the calibration, extend the comparison. Seeds 42, 79, and 123 were also used to choose the learning rate and step count.

**Results: single-shot, 35 seeds.** Random reaches held-out log-loss 0.600, 0.494, 0.406, 0.356 and AUC 0.883, 0.902, 0.921, 0.934 at 10, 20, 40, and 60 acquired groups (mean over 35 seeds); the 10 initial groups alone give log-loss 0.715 and AUC 0.872, so the labels teach the held-out preferences and the endpoint responds to the budget. The strategies differ overall (Friedman test across the 33 strategy variants with data on all 35 seeds: log-loss p = 0.0001, calibrated log-loss p < 0.0001, AUC p = 0.0001, accuracy p = 0.001). The table gives each strategy's gain over Random, averaged over budgets within a seed (positive = better, in log-loss units for the first two columns and AUC units for the third), with the Wilcoxon signed-rank p-value Holm-corrected across the strategies, and the fraction of seeds in which the strategy beat Random on log-loss. Full per-budget tables and the unadjusted p-values are in `results/pair_endpoint_study/single_all_*`.

| Strategy | log-loss gain | Holm p | cal. log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fisher D-optimal (Active Reward Modeling) | +0.041 | 0.155 | +0.009 | 0.717 | +0.008 | 0.909 | 66% (35) |
| BALD x P(decisive) | +0.038 | 0.026 | +0.008 | 0.877 | +0.010 | 0.027 | 77% (35) |
| Cluster-quota uncertainty | +0.038 | 0.033 | +0.003 | 1.000 | +0.005 | 1.000 | 74% (35) |
| Laplace BALD | +0.038 | 0.022 | +0.024 | 0.289 | +0.010 | 0.031 | 80% (35) |
| Core-set, relation-aware pairs | +0.036 | 0.176 | +0.038 | 0.067 | +0.006 | 1.000 | 74% (35) |
| Cluster-Margin | +0.033 | 0.285 | +0.026 | 0.481 | +0.002 | 1.000 | 69% (35) |
| MaxHerding (pairs) | +0.031 | 0.122 | +0.014 | 0.335 | +0.007 | 0.601 | 69% (35) |
| Cluster-quota uncertainty, original code, all heads | +0.029 | 0.176 | +0.002 | 1.000 | +0.001 | 1.000 | 74% (35) |
| DPP (quality x diversity) | +0.028 | 0.047 | +0.012 | 0.877 | +0.005 | 1.000 | 77% (35) |
| FASS (pairs) | +0.025 | 1.000 | +0.012 | 1.000 | +0.003 | 1.000 | 60% (35) |
| Core-set | +0.023 | 1.000 | +0.008 | 1.000 | -0.002 | 1.000 | 60% (35) |
| Graph facility location (uncertainty-weighted) | +0.020 | 1.000 | +0.003 | 1.000 | +0.004 | 1.000 | 51% (35) |
| MC-dropout variance | +0.019 | 1.000 | +0.003 | 1.000 | +0.003 | 1.000 | 57% (35) |
| Cluster-Margin, original code, all heads | +0.018 | 1.000 | +0.009 | 1.000 | -0.003 | 1.000 | 60% (35) |
| TypiClust (pairs) | +0.018 | 1.000 | +0.036 | 0.294 | +0.006 | 1.000 | 57% (35) |
| Largest predicted gap | +0.017 | 1.000 | +0.035 | 0.105 | +0.005 | 1.000 | 60% (35) |
| BADGE (pairs) | +0.016 | 1.000 | +0.009 | 1.000 | +0.002 | 1.000 | 60% (35) |
| ProbCover (pairs) | +0.016 | 1.000 | +0.023 | 0.330 | +0.005 | 1.000 | 66% (35) |
| Uncertainty + diversity | +0.015 | 1.000 | +0.013 | 1.000 | +0.001 | 1.000 | 63% (35) |
| Deep-ensemble BALD (8 heads) | +0.013 | 1.000 | +0.005 | 1.000 | +0.005 | 1.000 | 54% (35) |
| MC-dropout mutual info | +0.012 | 1.000 | +0.008 | 1.000 | +0.004 | 1.000 | 54% (35) |
| Image-coverage uncertainty | +0.010 | 1.000 | -0.014 | 1.000 | -0.000 | 1.000 | 63% (35) |
| Uncertainty | +0.007 | 1.000 | +0.000 | 1.000 | +0.001 | 1.000 | 51% (35) |
| Deep-ensemble BALD x P(decisive) | +0.004 | 1.000 | -0.002 | 1.000 | +0.005 | 1.000 | 51% (35) |
| Uncertainty, original code, all heads | +0.003 | 1.000 | -0.028 | 0.474 | -0.004 | 1.000 | 60% (35) |
| Uncertainty, all heads | +0.003 | 1.000 | -0.028 | 0.474 | -0.004 | 1.000 | 60% (35) |
| Uncertainty + diversity, original code, all heads | +0.002 | 1.000 | -0.014 | 1.000 | -0.006 | 1.000 | 57% (35) |
| Gap + posterior std (DeltaUCB-style) | -0.001 | 1.000 | +0.029 | 0.579 | +0.001 | 1.000 | 57% (35) |
| MC-dropout mutual info, original code, all heads | -0.015 | 1.000 | -0.020 | 1.000 | -0.007 | 1.000 | 54% (35) |
| DropQuery (pairs) | -0.016 | 1.000 | -0.019 | 0.483 | -0.004 | 1.000 | 37% (35) |
| MC-dropout variance, original code, all heads | -0.019 | 1.000 | -0.024 | 1.000 | -0.007 | 1.000 | 49% (35) |
| Graph cut (pairs) | -0.058 | 0.047 | -0.010 | 1.000 | -0.000 | 1.000 | 31% (35) |

Only **Laplace BALD** and **BALD × P(decisive)** beat Random on both log-loss and AUC after correction (Holm p = 0.022 and 0.026 on log-loss, 0.031 and 0.027 on AUC). Two further strategies are significant on log-loss alone, the original Cluster-quota uncertainty (Holm p = 0.033) and the DPP selector (0.047); graph cut is significantly *worse* than Random (−0.058, p = 0.047). On calibrated log-loss no strategy survives correction (the best, relation-aware core-set, has Holm p = 0.067), so part of the raw log-loss gain comes from the scale of the scores rather than from a better ordering of the pairs. The effects are small: Laplace BALD lowers log-loss from 0.600 to 0.542 at 10 groups and from 0.356 to 0.324 at 60, and raises AUC by about 0.01, and the gains are largest at the smallest budgets. Pure uncertainty rules do not help: Uncertainty (+0.007), MC-dropout mutual information (+0.012), and the all-head entropy rule (+0.003) are indistinguishable from Random, and DropQuery (−0.016) is no better. Deep-ensemble BALD, an alternative estimate of the same epistemic quantity, gains only +0.013 (Holm p = 1.0) and weighting it by P(decisive) +0.004; a plausible reason, which we did not test, is that heads trained to convergence on the same labels from different initialisations end up close to one another, leaving little disagreement to exploit, whereas the Laplace posterior reflects how little data constrains each direction of the last layer. The all-head versions of the original rules (rows marked `original code, all heads`) are all weaker than the originals: Cluster-quota uncertainty falls from +0.038 to +0.029 and is no longer significant (Holm p = 0.16), Cluster-Margin from +0.033 to +0.018, and the two MC-dropout rules from +0.019 and +0.012 to −0.019 and −0.015. The original Cluster-quota rule's advantage therefore depends on scoring only the head of the first judgment row rather than all heads. If the system chooses which type to ask for, as in the intended labeling protocol, the original scoring is legitimate and the all-head versions are a conservative alternative; both are reported. The five-seed ranking that first suggested a large advantage for Cluster-Margin (+0.110) and core-set (+0.076) did not replicate: over 35 seeds they are +0.033 and +0.023 and neither is significant after correction, an instance of the noise that a five-seed, 28-image comparison cannot exclude.

**Results: sequential, 35 seeds.** With rounds of 10 groups and retraining between rounds, Random reaches log-loss 0.578, 0.474, 0.398, 0.354 and AUC 0.891, 0.907, 0.922, 0.933 at 10, 20, 40, and 60 acquired groups. The strategies again differ overall (Friedman across the 33 variants: log-loss p = 0.001, calibrated log-loss p < 0.0001, AUC p = 0.004, accuracy p = 0.021), but **no strategy is significantly better than Random on any of the four metrics after Holm correction**, and the gains are smaller than in the single-shot condition:

| Strategy | log-loss gain | Holm p | cal. log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core-set, relation-aware pairs | +0.023 | 1.000 | +0.027 | 0.522 | +0.002 | 1.000 | 66% (35) |
| MaxHerding (pairs) | +0.021 | 1.000 | +0.008 | 1.000 | +0.004 | 1.000 | 66% (35) |
| BALD x P(decisive) | +0.017 | 1.000 | +0.004 | 1.000 | +0.004 | 1.000 | 60% (35) |
| Laplace BALD | +0.016 | 1.000 | +0.013 | 1.000 | +0.004 | 1.000 | 57% (35) |
| Cluster-quota uncertainty | +0.015 | 1.000 | +0.001 | 1.000 | -0.002 | 1.000 | 60% (35) |
| FASS (pairs) | +0.013 | 1.000 | +0.003 | 1.000 | -0.001 | 1.000 | 66% (35) |
| Fisher D-optimal (Active Reward Modeling) | +0.013 | 1.000 | -0.015 | 1.000 | +0.002 | 1.000 | 54% (35) |
| Cluster-Margin | +0.012 | 1.000 | +0.001 | 1.000 | -0.005 | 1.000 | 60% (35) |
| Core-set | +0.010 | 1.000 | -0.004 | 1.000 | -0.005 | 1.000 | 54% (35) |
| Cluster-Margin, original code, all heads | +0.010 | 1.000 | -0.011 | 1.000 | -0.004 | 1.000 | 54% (35) |
| Uncertainty + diversity, original code, all heads | +0.001 | 1.000 | -0.020 | 1.000 | -0.005 | 1.000 | 49% (35) |
| Largest predicted gap | +0.001 | 1.000 | +0.019 | 1.000 | +0.001 | 1.000 | 49% (35) |
| MC-dropout variance | -0.000 | 1.000 | -0.008 | 1.000 | -0.001 | 1.000 | 63% (35) |
| Uncertainty + diversity | -0.000 | 1.000 | -0.003 | 1.000 | -0.002 | 1.000 | 51% (35) |
| Graph facility location (uncertainty-weighted) | -0.000 | 1.000 | -0.007 | 1.000 | -0.001 | 1.000 | 51% (35) |
| DPP (quality x diversity) | -0.002 | 1.000 | +0.009 | 1.000 | -0.002 | 1.000 | 46% (35) |
| ProbCover (pairs) | -0.004 | 1.000 | +0.008 | 1.000 | +0.002 | 1.000 | 57% (35) |
| Cluster-quota uncertainty, original code, all heads | -0.005 | 1.000 | -0.016 | 1.000 | -0.007 | 0.452 | 51% (35) |
| TypiClust (pairs) | -0.007 | 1.000 | +0.019 | 1.000 | +0.001 | 1.000 | 46% (35) |
| Image-coverage uncertainty | -0.007 | 1.000 | -0.011 | 1.000 | -0.004 | 1.000 | 54% (35) |
| BADGE (pairs) | -0.009 | 1.000 | +0.002 | 1.000 | -0.001 | 1.000 | 49% (35) |
| Deep-ensemble BALD (8 heads) | -0.009 | 1.000 | -0.000 | 1.000 | -0.001 | 1.000 | 51% (35) |
| MC-dropout mutual info | -0.009 | 1.000 | -0.021 | 1.000 | -0.003 | 1.000 | 54% (35) |
| Uncertainty | -0.011 | 1.000 | -0.010 | 1.000 | -0.005 | 1.000 | 43% (35) |
| Gap + posterior std (DeltaUCB-style) | -0.020 | 1.000 | +0.001 | 1.000 | -0.005 | 1.000 | 37% (35) |
| Deep-ensemble BALD x P(decisive) | -0.021 | 1.000 | -0.001 | 1.000 | -0.000 | 1.000 | 40% (35) |
| Uncertainty, original code, all heads | -0.023 | 1.000 | -0.034 | 0.145 | -0.009 | 0.145 | 49% (35) |
| Uncertainty, all heads | -0.023 | 1.000 | -0.034 | 0.145 | -0.009 | 0.145 | 49% (35) |
| MC-dropout variance, original code, all heads | -0.025 | 1.000 | -0.040 | 1.000 | -0.009 | 1.000 | 54% (35) |
| MC-dropout mutual info, original code, all heads | -0.032 | 1.000 | -0.019 | 1.000 | -0.010 | 1.000 | 46% (35) |
| DropQuery (pairs) | -0.039 | 0.491 | -0.023 | 1.000 | -0.008 | 1.000 | 37% (35) |
| Graph cut (pairs) | -0.049 | 0.015 | -0.025 | 1.000 | -0.001 | 1.000 | 23% (35) |

Laplace BALD's log-loss gain falls from +0.038 (single-shot) to +0.016, BALD × P(decisive) from +0.038 to +0.017, and Cluster-quota uncertainty from +0.038 to +0.015, none of them significant. The relation-aware core-set (+0.023) and MaxHerding (+0.021) lead, also not significant (Holm p = 1.0). Graph cut is again significantly worse than Random (−0.049, Holm p = 0.015), and the uncertainty-driven rules (Uncertainty −0.011, all-head entropy −0.023, DropQuery −0.039, DeltaUCB-style −0.020) are at or below Random; the all-head MC-dropout variants (−0.025, −0.032) and all-head Uncertainty (−0.023) are lower still. Retraining between rounds therefore did not help the model-based rules; the one effect that appears in both conditions is that graph cut is harmful.

**Which strategy at which budget.** The tables above average each seed's gain over the four budgets, which gives the test its power but hides differences between budgets. The question of which rule is best at which budget is answered separately here: for each condition, metric, and budget, the per-seed gain over Random is tested with a Wilcoxon signed-rank test over the 35 seeds, Holm-corrected across the 32 strategy variants within that cell (`analyze_by_budget.py`; `results/pair_endpoint_study/by_budget_gain_vs_random.csv`). Gains are in log-loss units (positive = lower loss than Random) or AUC units (positive = higher AUC).

*Single-shot* (one batch of the budget size chosen from the 10-group model):

| Budget | Metric | Three largest gains over Random (Holm p) | Significant after Holm correction within this cell |
|---:|---|---|---|
| 10 | log-loss | Cluster-quota uncertainty +0.082 (0.06); Cluster-quota uncertainty, original code, all heads +0.077 (0.04); BALD x P(decisive) +0.071 (0.10) | Cluster-quota uncertainty, original code, all heads +0.077; DPP (quality x diversity) +0.067 |
| 10 | AUC | MaxHerding (pairs) +0.018 (0.27); BALD x P(decisive) +0.017 (0.14); DPP (quality x diversity) +0.016 (0.24) | none |
| 20 | log-loss | Core-set, relation-aware pairs +0.061 (0.20); Core-set +0.053 (0.62); Cluster-Margin +0.046 (0.20) | none |
| 20 | AUC | BALD x P(decisive) +0.011 (0.74); MaxHerding (pairs) +0.010 (0.90); Largest predicted gap +0.009 (1.00) | none |
| 40 | log-loss | Fisher D-optimal (Active Reward Modeling) +0.052 (0.00); BALD x P(decisive) +0.039 (0.00); BADGE (pairs) +0.037 (0.22) | Fisher D-optimal (Active Reward Modeling) +0.052; BALD x P(decisive) +0.039; Graph facility location (uncertainty-weighted) +0.029 |
| 40 | AUC | BALD x P(decisive) +0.011 (0.08); Fisher D-optimal (Active Reward Modeling) +0.010 (0.10); BADGE (pairs) +0.009 (1.00) | none |
| 60 | log-loss | Laplace BALD +0.032 (0.01); FASS (pairs) +0.017 (1.00); DPP (quality x diversity) +0.016 (0.93) | Laplace BALD +0.032 |
| 60 | AUC | Laplace BALD +0.012 (0.00); DPP (quality x diversity) +0.005 (1.00); Gap + posterior std (DeltaUCB-style) +0.003 (1.00) | Laplace BALD +0.012 |

*Sequential* (rounds of 10 groups):

| Budget | Metric | Three largest gains over Random (Holm p) | Significant after Holm correction within this cell |
|---:|---|---|---|
| 10 | log-loss | Cluster-quota uncertainty +0.060 (1.00); Cluster-quota uncertainty, original code, all heads +0.055 (0.61); BALD x P(decisive) +0.049 (1.00) | Graph cut (pairs) -0.120 |
| 10 | AUC | MaxHerding (pairs) +0.010 (1.00); BALD x P(decisive) +0.008 (1.00); DPP (quality x diversity) +0.007 (1.00) | none |
| 20 | log-loss | Core-set, relation-aware pairs +0.041 (1.00); Core-set +0.033 (1.00); MaxHerding (pairs) +0.016 (1.00) | none |
| 20 | AUC | Core-set, relation-aware pairs +0.004 (1.00); MaxHerding (pairs) +0.003 (1.00); Laplace BALD +0.003 (1.00) | none |
| 40 | log-loss | FASS (pairs) +0.034 (0.05); Fisher D-optimal (Active Reward Modeling) +0.025 (1.00); Graph facility location (uncertainty-weighted) +0.025 (1.00) | FASS (pairs) +0.034 |
| 40 | AUC | FASS (pairs) +0.008 (0.34); Graph facility location (uncertainty-weighted) +0.006 (1.00); Core-set, relation-aware pairs +0.005 (1.00) | none |
| 60 | log-loss | Largest predicted gap +0.024 (0.03); Image-coverage uncertainty +0.023 (0.10); MC-dropout mutual info, original code, all heads +0.023 (0.03) | Largest predicted gap +0.024; MC-dropout mutual info, original code, all heads +0.023 |
| 60 | AUC | Largest predicted gap +0.008 (0.05); MC-dropout mutual info, original code, all heads +0.006 (0.38); BALD x P(decisive) +0.005 (1.00) | Largest predicted gap +0.008 |

In the single-shot condition the leaders move with the budget: at 10 groups the leading rules are Cluster-quota uncertainty (original +0.082, all-head +0.077, the latter significant) and DPP (+0.067), at 20 the relation-aware and plain core-set, at 40 the Fisher D-optimal and BALD × P(decisive) rules, and at 60 Laplace BALD (log-loss +0.032, AUC +0.012). This looks like coverage- or diversity-oriented rules doing better when few groups can be labelled and rules built on the model's posterior doing better as the budget grows, which is the direction of the low-budget-versus-high-budget argument of Hacohen et al. (2022; notes, third round). It is a plausible pattern, not a conclusion, for three reasons. (i) Each cell reports the best of 32 variants, which overstates the best one. (ii) Holm correction is applied within each cell only, not across budgets, metrics, and conditions, so the number of significant cells is larger than a family-wise correction over all 32 × 4 × 2 × 2 comparisons would leave. (iii) The sequential condition does not reproduce the pattern: its leaders are Cluster-quota uncertainty (not significant) at 10, relation-aware core-set (not significant) at 20, FASS (+0.034, significant on log-loss only) at 40, and the largest-predicted-gap rule (+0.024 log-loss, +0.008 AUC) and all-head MC-dropout mutual information at 60. In addition, a budget of 60 groups is 78–94% of the candidate pool (64–77 groups), so at that budget every strategy selects most of the pool and the strategies differ only in the 4–17 groups they leave out; the budget-60 results should not be read as evidence about large budgets. Establishing the pattern would need a larger candidate pool and a correction over all cells.

**Sensitivity to the training schedule.** To check that the single-shot results do not hinge on the calibrated learning rate and step count, we repeated the single-shot comparison for 13 strategies and Random on the same 35 seeds with learning rate 0.003 and 300 steps instead of 0.01 and 100 (`results/pair_endpoint_study/single_all_cells_alt_lr003_300steps_*`). The strategies still differ overall on log-loss (Friedman p < 0.0001) and calibrated log-loss (p = 0.019) but not on AUC (p = 0.14) or accuracy (p = 0.59). After Holm correction across these 14 variants, the significant log-loss gains are now the relation-aware core-set (+0.063, p = 0.003) and the original Cluster-quota uncertainty (+0.051, p = 0.004), graph cut is again significantly worse (−0.066, p = 0.005), and Laplace BALD (+0.044, p = 0.091) and BALD × P(decisive) (+0.018, p = 0.53) are no longer significant. On AUC only Laplace BALD is significant (+0.007, p = 0.022; +0.010 under the calibrated schedule). So the log-loss ranking of the information-based and coverage-based rules moves with the training schedule, which is consistent with log-loss depending on the scale of the scores, whereas three statements hold under both schedules: Laplace BALD has a small positive AUC gain, graph cut is worse than Random, and no uncertainty-driven rule beats Random.

**Cold start.** Replacing the study's default initial set (a random draw with greedy reconstruction-type coverage) by label-free choices (farthest-first, k-means, TypiClust in the relation-aware pair space, or a random draw) changes the initial model's AUC by at most 0.012 (35 seeds; paired Wilcoxon p ≥ 0.25 against the default). The raw log-loss of the initial model is lower for farthest-first and k-means (0.567 and 0.599 versus 0.715), but calibrated log-loss is the same for all sets, so this difference reflects score scale rather than ranking quality. After 30 further random groups all initial sets are within noise (`notes/literature_and_new_strategy_ideas.md`). A first analysis on five seeds had suggested that the type-covering default is better; that did not replicate and is not claimed.

![Gain over random selection on the held-out preference endpoint](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/endpoint_gain_over_random.png)

*Figure 19. Gain of each of 33 acquisition strategy variants over Random on held-out preference prediction, 35 seeds, single-shot (top) and sequential (bottom) acquisition; left: log-loss gain, right: AUC gain (positive = better than Random). Points are means of the per-seed gain averaged over budgets of 10–60 groups; bars are 95% bootstrap intervals over seeds; filled points are significant by a Wilcoxon signed-rank test with Holm correction across strategies. Strategies are ordered by their single-shot log-loss gain.*

**Reading the two conditions together.** On a preference endpoint that the labels can move, the choice among 33 acquisition-rule variants matters little at this scale. The strongest evidence is for a small benefit of information-based rules built on the last-layer Fisher/Laplace posterior (Laplace BALD, BALD × P(decisive)) when one batch is chosen from the 10-group model, of which only Laplace BALD keeps a significant AUC gain under a second training schedule: log-loss is lower by about 0.04 and AUC higher by about 0.01 at budgets of 10–60 groups, significant after correction within that condition but not reproduced with sequential retraining, so it should be treated as suggestive. The finding that is consistent across conditions is negative: uncertainty-driven selection (including MC-dropout, BADGE-style gradient embeddings, and DropQuery) is no better than random choice, and graph cut is worse. Differences between all other rules are below what 35 seeds can resolve.

**Limitations.** (i) The endpoint is prediction of held-out *human* judgments in this data set, with 51–79 decisive test rows per seed (mean 63); it is not the downstream type classification of §5.11, which the anchors saturate. (ii) The eight original strategies score only the head of the type of each group's first judgment row, treated as the queried type, yet acquire every judgment of the selected group; the all-head versions score all four heads instead. A design in which the query unit is a (pair, type) judgment, with the budget counted in judgments, would remove this mismatch and has not been run. The all-head versions apply the original code to a model whose Twinned head is silenced rather than re-deriving each rule for four heads. (iii) Hyper-parameters were tuned on random batches (learning rate 0.01, 100 steps), not for each strategy, and a second schedule (learning rate 0.003, 300 steps) changes which strategies are significant on log-loss (see above), so log-loss rankings depend on the schedule. (iv) The per-seed gain is averaged over four budgets, so a strategy that helps only at one budget is diluted. (v) Seeds 42, 79, and 123 informed the hyper-parameter choice, and Holm correction is applied within each condition, not across the two. (vi) The strategies include implementations of published methods adapted to a Bradley–Terry head (e.g. TypiClust, BADGE, FASS, ProbCover); we did not tune their own hyper-parameters, and a different adaptation could change their ranking.





### 5.13 A second split following classifier2: 20% of pair groups held out

**Why a second split.** The split of §5.12 holds out 40 test and 20 validation groups, so the evaluation sets are as large as the largest training set (10 initial + 60 acquired groups) and the candidate pool is only 64–77 groups. The results of §5.12 are therefore results for a very small labelled budget evaluated on a comparatively large held-out set. To test them under the data-splitting practice of the earlier classifier2 work (`classifier2/classifier_training_code/train_unified.py`, `load_data`: shuffle the unique image pairs and hold out 20%, all judgments of a pair on the same side, no separate validation set), the comparison is repeated with that split.

**Split.** For each of the same 35 seeds, 20% of the 168 usable pair groups (34) are held out as the test set, every group that shares an image with a test group is removed from training, and the 10 initial groups are drawn from the remainder by the study's rule (random with greedy coverage of the reconstruction types). The remaining 99–113 groups form the candidate pool, so the largest budget (60 groups) is 53–61% of the pool. The test set holds 35–67 decisive judgments per seed (mean 53.5). There is no validation set, so the learning rate and step count of §5.12 (0.01, 100 steps, chosen on different groups) are reused without recalibration, and the temperature-calibrated log-loss is not reported. Strategies, budgets (10, 20, 40, 60), the single-shot and sequential conditions, and the statistics are those of §5.12 (`PAIR_STUDY_SPLIT=classifier2`).

**Results: single-shot, 35 seeds.** Random reaches log-loss 0.574, 0.484, 0.392, 0.359 and AUC 0.886, 0.905, 0.924, 0.930 at 10, 20, 40, and 60 acquired groups; the 10 initial groups give log-loss 0.667 and AUC 0.865. The strategies differ overall (Friedman test across 33 variants, p < 0.0001 for log-loss, AUC, and accuracy), but after Holm correction **only plain core-set** is significantly better than Random, on log-loss (+0.052, p = 0.012, better in 77% of seeds); no strategy is significant on AUC or accuracy.

| Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |
|---|---:|---:|---:|---:|---:|
| Core-set | +0.052 | 0.012 | +0.003 | 1.000 | 77% (35) |
| Core-set, relation-aware pairs | +0.042 | 0.391 | +0.004 | 1.000 | 60% (35) |
| BALD x P(decisive) | +0.037 | 0.314 | +0.008 | 0.654 | 71% (35) |
| Gap + posterior std (DeltaUCB-style) | +0.027 | 1.000 | +0.001 | 1.000 | 63% (35) |
| Laplace BALD | +0.027 | 1.000 | +0.003 | 1.000 | 66% (35) |
| ProbCover (pairs) | +0.025 | 1.000 | +0.005 | 1.000 | 60% (35) |
| TypiClust (pairs) | +0.024 | 1.000 | +0.006 | 1.000 | 57% (35) |
| Deep-ensemble BALD (8 heads) | +0.021 | 1.000 | +0.007 | 0.654 | 71% (35) |
| Graph facility location (uncertainty-weighted) | +0.020 | 1.000 | +0.004 | 1.000 | 74% (35) |
| Fisher D-optimal (Active Reward Modeling) | +0.020 | 1.000 | +0.002 | 1.000 | 69% (35) |
| FASS (pairs) | +0.016 | 1.000 | +0.000 | 1.000 | 60% (35) |
| Largest predicted gap | +0.015 | 1.000 | -0.001 | 1.000 | 60% (35) |
| Cluster-Margin | +0.014 | 1.000 | -0.003 | 1.000 | 57% (35) |
| MaxHerding (pairs) | +0.014 | 1.000 | +0.004 | 1.000 | 66% (35) |
| BADGE (pairs) | +0.012 | 1.000 | +0.005 | 1.000 | 63% (35) |
| Deep-ensemble BALD x P(decisive) | +0.012 | 1.000 | +0.003 | 1.000 | 60% (35) |
| Cluster-Margin, all heads | +0.012 | 1.000 | -0.008 | 1.000 | 49% (35) |
| Uncertainty + diversity, all heads | -0.001 | 1.000 | -0.007 | 0.654 | 46% (35) |
| DropQuery (pairs) | -0.001 | 1.000 | -0.006 | 1.000 | 46% (35) |
| Uncertainty + diversity | -0.002 | 1.000 | -0.001 | 1.000 | 54% (35) |
| MC-dropout mutual info, all heads | -0.006 | 1.000 | -0.005 | 1.000 | 51% (35) |
| Image-coverage uncertainty | -0.008 | 1.000 | -0.010 | 0.091 | 40% (35) |
| Cluster-quota uncertainty, all heads | -0.010 | 1.000 | -0.011 | 0.219 | 49% (35) |
| Uncertainty, all heads | -0.012 | 1.000 | -0.013 | 0.175 | 46% (35) |
| Uncertainty, all heads | -0.012 | 1.000 | -0.013 | 0.175 | 46% (35) |
| Uncertainty | -0.012 | 1.000 | -0.007 | 0.493 | 34% (35) |
| Cluster-quota uncertainty | -0.015 | 1.000 | -0.008 | 1.000 | 43% (35) |
| DPP (quality x diversity) | -0.016 | 1.000 | -0.007 | 1.000 | 49% (35) |
| MC-dropout mutual info | -0.018 | 1.000 | -0.006 | 0.819 | 43% (35) |
| MC-dropout variance, all heads | -0.020 | 1.000 | -0.007 | 1.000 | 46% (35) |
| MC-dropout variance | -0.029 | 1.000 | -0.009 | 0.183 | 29% (35) |
| Graph cut (pairs) | -0.048 | 0.909 | -0.002 | 1.000 | 46% (35) |

The information-based rules that were significant in §5.12 are not here: Laplace BALD +0.027 (Holm p = 1.0), BALD × P(decisive) +0.037 (p = 0.31), Fisher D-optimal +0.020 (p = 1.0). The uncertainty-driven rules are again no better than Random and mostly worse (Uncertainty −0.012, Cluster-quota uncertainty −0.015, MC-dropout mutual information −0.018, MC-dropout variance −0.029; none significant), and graph cut keeps its negative sign (−0.048) but is no longer significant (p = 0.91).

**By budget.** The same per-budget analysis as in §5.12 (single-shot, Holm correction within each cell):

| Budget | Metric | Three largest gains over Random (Holm p) | Significant after Holm correction within this cell |
|---:|---|---|---|
| 10 | log-loss | Graph facility location (uncertainty-weighted) +0.072 (0.01); Gap + posterior std (DeltaUCB-style) +0.071 (0.85); Core-set, relation-aware pairs +0.069 (0.70) | Graph facility location (uncertainty-weighted) +0.072 |
| 10 | AUC | Deep-ensemble BALD (8 heads) +0.018 (0.00); TypiClust (pairs) +0.016 (0.46); BALD x P(decisive) +0.013 (0.12) | Deep-ensemble BALD (8 heads) +0.018 |
| 20 | log-loss | Core-set +0.080 (0.01); BALD x P(decisive) +0.050 (1.00); Deep-ensemble BALD (8 heads) +0.044 (1.00) | Core-set +0.080 |
| 20 | AUC | BALD x P(decisive) +0.012 (1.00); Deep-ensemble BALD (8 heads) +0.009 (1.00); Core-set +0.008 (1.00) | none |
| 40 | log-loss | Core-set +0.047 (0.11); Core-set, relation-aware pairs +0.037 (0.93); Laplace BALD +0.034 (0.80) | none |
| 40 | AUC | Laplace BALD +0.006 (1.00); Core-set +0.004 (1.00); Core-set, relation-aware pairs +0.004 (1.00) | none |
| 60 | log-loss | Core-set, relation-aware pairs +0.031 (0.57); Laplace BALD +0.027 (0.42); FASS (pairs) +0.023 (0.11) | none |
| 60 | AUC | Core-set, relation-aware pairs +0.008 (1.00); FASS (pairs) +0.007 (0.38); Laplace BALD +0.005 (1.00) | none |

Coverage-oriented rules lead at the smallest budgets: graph facility location (+0.072 log-loss) at 10 groups and core-set (+0.080) at 20, both significant, and deep-ensemble BALD has a significant AUC gain at 10 (+0.018). At 40 and 60 groups nothing is significant (core-set +0.047 at 40, p = 0.11; relation-aware core-set +0.031 at 60). The lowest budgets favouring coverage is in line with §5.12 (where Cluster-quota uncertainty and DPP led at 10), but the leading rule differs between the two splits, and each cell is again the best of 32 variants with correction only within the cell.

**Results: sequential, 35 seeds.** With rounds of 10 groups on this split, Random reaches log-loss 0.560, 0.481, 0.407, 0.366 and AUC 0.886, 0.902, 0.918, 0.928 at 10, 20, 40, and 60 acquired groups. The strategies differ overall (Friedman: p < 0.0001 for log-loss, AUC, and accuracy), and again **only core-set** is significantly better than Random after correction, on log-loss (+0.053, Holm p = 0.009); no strategy is significant on AUC or accuracy. Core-set is therefore the one rule whose advantage appears under both acquisition conditions on this split (+0.052 single-shot, +0.053 sequential).

| Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |
|---|---:|---:|---:|---:|---:|
| Core-set | +0.053 | 0.009 | +0.006 | 1.000 | 71% (35) |
| Core-set, relation-aware pairs | +0.044 | 0.424 | +0.006 | 1.000 | 69% (35) |
| BALD x P(decisive) | +0.043 | 0.877 | +0.011 | 0.114 | 69% (35) |
| Gap + posterior std (DeltaUCB-style) | +0.041 | 0.922 | +0.005 | 1.000 | 60% (35) |
| Largest predicted gap | +0.038 | 1.000 | +0.007 | 1.000 | 63% (35) |
| Cluster-Margin, original code, all heads | +0.037 | 0.498 | +0.001 | 1.000 | 74% (35) |
| Laplace BALD | +0.037 | 1.000 | +0.006 | 1.000 | 66% (35) |
| ProbCover (pairs) | +0.032 | 1.000 | +0.008 | 0.380 | 66% (35) |
| Graph facility location (uncertainty-weighted) | +0.032 | 0.969 | +0.006 | 1.000 | 66% (35) |
| Deep-ensemble BALD (8 heads) | +0.031 | 1.000 | +0.009 | 0.147 | 63% (35) |
| Deep-ensemble BALD x P(decisive) | +0.025 | 1.000 | +0.007 | 0.565 | 69% (35) |
| Cluster-Margin | +0.022 | 1.000 | +0.000 | 1.000 | 66% (35) |
| Fisher D-optimal (Active Reward Modeling) | +0.019 | 1.000 | +0.006 | 1.000 | 69% (35) |
| BADGE (pairs) | +0.019 | 0.870 | +0.007 | 0.274 | 71% (35) |
| TypiClust (pairs) | +0.015 | 1.000 | +0.009 | 0.689 | 46% (35) |
| MaxHerding (pairs) | +0.015 | 1.000 | +0.006 | 1.000 | 54% (35) |
| Uncertainty + diversity, original code, all heads | +0.010 | 1.000 | -0.001 | 1.000 | 66% (35) |
| Cluster-quota uncertainty, original code, all heads | +0.001 | 1.000 | -0.006 | 1.000 | 57% (35) |
| Cluster-quota uncertainty | -0.003 | 1.000 | -0.000 | 1.000 | 57% (35) |
| MC-dropout mutual info, original code, all heads | -0.006 | 1.000 | -0.004 | 1.000 | 43% (35) |
| FASS (pairs) | -0.007 | 1.000 | -0.005 | 1.000 | 51% (35) |
| Image-coverage uncertainty | -0.011 | 1.000 | -0.007 | 0.277 | 43% (35) |
| Uncertainty + diversity | -0.012 | 1.000 | -0.002 | 1.000 | 49% (35) |
| MC-dropout variance | -0.014 | 1.000 | -0.002 | 1.000 | 49% (35) |
| MC-dropout mutual info | -0.018 | 1.000 | -0.004 | 1.000 | 40% (35) |
| DropQuery (pairs) | -0.018 | 1.000 | -0.008 | 0.359 | 43% (35) |
| Uncertainty | -0.021 | 1.000 | -0.006 | 1.000 | 60% (35) |
| DPP (quality x diversity) | -0.024 | 1.000 | -0.005 | 1.000 | 46% (35) |
| Uncertainty, original code, all heads | -0.026 | 0.922 | -0.011 | 0.102 | 34% (35) |
| Uncertainty, all heads | -0.026 | 0.922 | -0.011 | 0.102 | 34% (35) |
| Graph cut (pairs) | -0.027 | 1.000 | +0.004 | 1.000 | 54% (35) |
| MC-dropout variance, original code, all heads | -0.029 | 1.000 | -0.009 | 0.179 | 34% (35) |

Laplace BALD (+0.037, Holm p = 1.0), BALD × P(decisive) (+0.043, p = 0.88), and deep-ensemble BALD (+0.031, p = 1.0) are positive but not significant on log-loss, and the uncertainty-driven rules are again at or below Random (Uncertainty −0.021, MC-dropout mutual information −0.018). Per budget (sequential, correction within each cell):

| Budget | Metric | Three largest gains over Random (Holm p) | Significant after Holm correction within this cell |
|---:|---|---|---|
| 10 | log-loss | Graph facility location (uncertainty-weighted) +0.058 (0.36); Gap + posterior std (DeltaUCB-style) +0.057 (0.97); Core-set, relation-aware pairs +0.055 (1.00) | none |
| 10 | AUC | Deep-ensemble BALD (8 heads) +0.017 (0.05); TypiClust (pairs) +0.014 (1.00); BALD x P(decisive) +0.012 (1.00) | Deep-ensemble BALD (8 heads) +0.017 |
| 20 | log-loss | Core-set +0.076 (0.02); Cluster-Margin, original code, all heads +0.065 (0.16); Laplace BALD +0.063 (0.88) | Core-set +0.076 |
| 20 | AUC | TypiClust (pairs) +0.011 (0.83); ProbCover (pairs) +0.011 (1.00); Deep-ensemble BALD (8 heads) +0.011 (1.00) | none |
| 40 | log-loss | Core-set +0.062 (0.00); Core-set, relation-aware pairs +0.053 (0.11); Cluster-Margin, original code, all heads +0.051 (0.15) | Core-set +0.062 |
| 40 | AUC | BALD x P(decisive) +0.015 (0.01); Cluster-Margin, original code, all heads +0.013 (0.21); Core-set +0.012 (0.34) | BALD x P(decisive) +0.015 |
| 60 | log-loss | Largest predicted gap +0.040 (0.02); Core-set, relation-aware pairs +0.039 (0.10); BALD x P(decisive) +0.032 (0.21) | Largest predicted gap +0.040 |
| 60 | AUC | Largest predicted gap +0.011 (0.01); Core-set, relation-aware pairs +0.009 (0.05); BALD x P(decisive) +0.008 (0.97) | Largest predicted gap +0.011; Core-set, relation-aware pairs +0.009 |

Core-set is significant at 20 and 40 groups (+0.076, +0.062 log-loss); the largest-predicted-gap rule is significant at 60 (+0.040 log-loss, +0.011 AUC), BALD × P(decisive) has a significant AUC gain at 40 (+0.015), and deep-ensemble BALD at 10 (+0.017 AUC, p = 0.05). Each of these is the best of 32 variants in its cell.

**What the two splits say together.** Two statements hold under both splits: no uncertainty-driven rule (Uncertainty, MC-dropout, BADGE-style, DropQuery) is significantly better than Random, and gains over Random are small (log-loss about 0.03–0.05, AUC at most 0.01). The statements that differ are the ones about which rules help: Laplace BALD and BALD × P(decisive) are significant under the small-pool split of §5.12 (single-shot only) and not under this one, while core-set is significant here under both acquisition conditions (+0.052, +0.053) and not in §5.12 (+0.023 single-shot, +0.010 sequential); no rule is significant under both splits. The ranking of coverage-type and information-type rules therefore depends on the split, plausibly on the size of the candidate pool and of the test set, and the data do not support recommending one rule. A larger candidate pool and a second encoder are needed to settle it.

### 5.14 Re-running the comparison with one (pair, type) judgment as the query unit

**Why.** In the labelling software an expert is shown one image pair and one reconstruction type and states which image wins (or tie, or not applicable), so the unit of one query is a single (pair, type) judgment. The studies of §5.10–5.13 instead selected a whole pair group and revealed all of its recorded judgments (3.1 on average, one per type), and counted the budget in groups. A budget of 10 groups therefore delivered about 31 judgments, the types of a selected group were revealed whether or not a strategy would have asked for them, and the pool held 64–77 (Split A) or 99–113 (Split B) candidates. This was a design flaw of the earlier simulation, and it applies to every strategy. The comparison was repeated with the unit of selection and the budget both counted in judgments (`JUDGMENT_UNIT_RESULTS.md`, `judgment_unit_*.py`).

**Design.** The splits, the 35 seeds, the held-out sets, the frozen features, the head and its schedule (lr 0.01, 100 steps), the endpoint and the statistics are those of §5.12–5.13 (imported, not copied). The candidate pool is every judgment of the pool groups (220 judgments on average in Split A, 329 in Split B); the hidden oracle reveals only the selected judgment's outcome, and an unselected judgment of a selected pair is not trained on (unit test). Budgets are 10, 20, 40 and 60 judgments, single-shot and sequential (rounds of 10), with the 10 initial groups' judgments (about 30) revealed as before. Budget 100 was not run: the pair-level rules choose one judgment per pair and Split A has only 64–77 pool pairs. Random is uniform over the remaining judgments (mean of 5 draws). A second baseline, a uniform pair followed by a uniform type, is statistically indistinguishable from Random in all four cells. All 33 variants were adapted to the new unit (the full table is in the note). Row-level rules (uncertainty and its variants, BALD, Fisher, ensemble BALD, gap rules) score each judgment with the head of its own type and so choose the type; pair-level rules (core-set, TypiClust, ProbCover, MaxHerding, graph cut, and the quality-weighted DPP, FASS and graph facility location) work on unique pairs and reveal one judgment per chosen pair, with a random type, or the most uncertain one for rules that carry an uncertainty term; the all-head variants choose the pair and reveal a random type. These adaptations are my own decisions and other reasonable ones exist.

**Result 1: almost nothing is significant on the primary endpoints.** Wilcoxon signed-rank tests of the per-seed gain over Random, averaged over budgets, with Holm correction over the 32 variants within each table, give the following.

| Split, condition | Variants better than Random (Holm < 0.05) | Variants worse than Random (Holm < 0.05) |
|---|---|---|
| Split A, single-shot | none (log-loss, calibrated log-loss, AUC) | none |
| Split A, sequential | none | none |
| Split B, single-shot | none | DPP (log-loss −0.104, better in 14% of seeds); all-head Uncertainty (AUC −0.026); all-head MC-dropout mutual information (AUC −0.014) |
| Split B, sequential | all-head Cluster-quota uncertainty (log-loss +0.074, Holm p < 0.001, better in 80% of seeds); deep-ensemble BALD × P(decisive) (AUC +0.014, p = 0.023) | none |

The one log-loss result is fragile: the same rule scoring only the queried type gains +0.039 (not significant), and the all-head rule is +0.016 (Split A sequential) and −0.005 (Split B single-shot), also not significant. Uncertainty-driven rules (Uncertainty, MC-dropout, BADGE, DropQuery) are again never better than Random.

**Result 2: the significant rules of §5.12–5.13 do not reproduce.**

| Split, condition | Strategy | Group unit: log-loss gain (Holm p) | Judgment unit: log-loss gain (Holm p, seeds better) |
|---|---|---|---|
| A, single-shot | Laplace BALD | +0.038 (0.022) | −0.001 (1.0, 51%) |
| A, single-shot | BALD × P(decisive) | +0.038 (0.026) | +0.004 (1.0, 49%) |
| A, single-shot | Cluster-quota uncertainty | +0.038 (0.033) | +0.014 (1.0, 60%) |
| B, single-shot | Core-set | +0.052 (0.012) | +0.029 (1.0, 60%) |
| B, sequential | Core-set | +0.053 (0.009) | +0.063 (0.27, 66%) |

Core-set on Split B keeps its sign and the graph-cut sign (worse, Split A) is retained, but neither is significant; the Laplace-BALD advantage of Split A disappears. The statement of §5.13 that no rule is significant under both splits therefore still holds, and which rule is significant now depends on the query unit as well as on the split. Absolute levels are not comparable: Random's log-loss at 60 judgments is 0.47–0.51 against 0.35–0.37 at 60 groups, and its AUC 0.90 against 0.93.

**Result 3: by budget.** As in §5.12, the best rules at each budget are the best of 32 and Holm applies only within a cell. Coverage-type rules lead the log-loss ranking at 10 and 20 judgments in Split A single-shot (relation-aware core-set +0.074 and +0.102, not significant), and only a few cells are significant: core-set at 60 judgments in Split A sequential (+0.078), and in Split B sequential relation-aware core-set (+0.114) and the all-head Cluster-quota rule (+0.106) at 40 and BALD × P(decisive) at 60 (log-loss +0.098, AUC +0.020). In Split B single-shot, DPP (−0.130 at 20) and plain Uncertainty (−0.060 at 60) are significantly worse than Random. The earlier impression that coverage rules help at small budgets and information rules at large ones is not clearly supported by this run either: the few significant cells are core-set and the all-head Cluster-quota rule at 40–60 judgments and BALD × P(decisive) at 60.

**Result 4: a secondary signal, treated as a hypothesis.** On decisive-judgment accuracy, which had no Holm-significant rule in the group-unit study, deep-ensemble BALD, deep-ensemble BALD × P(decisive) and Fisher D-optimal design are 1–2.5 points above Random. The effect is Holm-significant in three of the four main cells for deep-ensemble BALD × P(decisive) (+0.015 Split A single-shot, +0.020 Split A sequential, +0.022 Split B sequential) and in two for Fisher D-optimal (+0.024, +0.020, both sequential); these rules choose the type with the model, and their AUC gain is only +0.00 to +0.014. With a random initial set (below) nothing is significant (20 seeds, lower power), though the sign stays positive in all four cells (+0.006 to +0.022). Accuracy was not a pre-specified primary endpoint of the main run, and with four metrics, two splits and two conditions a few nominal hits are expected by chance; this is why it was tested on new seeds (below).

**Pre-registered replication on new seeds.** The hypotheses were written and committed before running (`JUDGMENT_UNIT_REPLICATION_PREREGISTRATION.md`) and tested on seeds 500–534 (new random splits and initial sets; the same 168 pair groups, so this checks sensitivity to the split draw and is not a test on new data; Holm over 18 one-sided tests; `JUDGMENT_UNIT_REPLICATION_RESULTS.md`). Results, with the main-run estimate in brackets:

| Hypothesis | New seeds | Outcome |
|---|---|---|
| Deep-ensemble BALD × P(decisive), AUC, Split B sequential | +0.014 (+0.014), Holm 0.019, 69% of seeds better | replicated |
| Decisive accuracy above Random for deep-ensemble BALD × P(decisive), deep-ensemble BALD and Fisher D-optimal in all four cells | all 12 estimates positive, +0.008 to +0.030; 11 of 12 pass Holm | replicated for BALD × P(decisive) (4/4) and Fisher (4/4), for ensemble BALD in 3/4 (Split A single-shot +0.008, Holm 0.24) |
| All-head Cluster-quota uncertainty, log-loss, Split B sequential | +0.044 (+0.074), Holm 0.110, 57% of seeds | not replicated |
| Core-set, log-loss, Split B | +0.002 and +0.029 (+0.029, +0.063), Holm 0.36 and 0.24 | not replicated |
| DPP worse than Random, log-loss, Split B single-shot | −0.035 (−0.104), Holm 0.24, 51% of seeds worse | not replicated |
| All-head Uncertainty worse than Random, AUC, Split B single-shot | −0.015 (−0.026), Holm 0.015 | replicated |
| Plain Uncertainty no better than Random | all 8 estimates negative, none significant | supported |

All 18 estimates have the predicted sign, but the log-loss effects shrink a lot (to a third or less for DPP and core-set), whereas the AUC and accuracy effects are stable (correlation of the 13 variants' mean accuracy gains between old and new seeds 0.64–0.89, of their log-loss gains 0.03–0.65). So the gain of two rules that choose the type with the model (deep-ensemble BALD × P(decisive), Fisher D-optimal) of about 1–3 points of decisive accuracy is not an accident of the 35 original splits, but it is modest, comes with no log-loss gain that survives, and has not been tested on new data. Pooling the 70 seeds is exploratory (it contains the seeds on which the hypotheses were formed).

**What the strategies reveal.** At 60 judgments, plain Uncertainty, Cluster-quota uncertainty and Cluster-Margin choose decisive judgments only 28–36% of the time, against 50% for Random, because the uncertain judgments are largely ties and not-applicable ones, which carry little preference information; the largest-predicted-gap rules and BALD × P(decisive) choose 59–69% decisive ones. Laplace BALD and BALD × P(decisive) concentrate the budget on 30–35 distinct pairs, several types of the same pair, whereas pair-level rules necessarily use 60 distinct pairs and Random about 42–48. This is one plausible reason why uncertainty rules do not help on this endpoint; it was not tested.

**Sensitivity to the head schedule.** The head schedule (lr 0.01, 100 steps) had been calibrated on random batches of the group-unit design. It was recalibrated for the judgment unit on Split A's validation groups only (seeds 42, 79, 123, selection by validation log-loss, outer test unused; `JUDGMENT_UNIT_RETUNED_RESULTS.md`): the best setting is the gentlest of the grid, lr 0.001 with 100 steps for the initial rows and 10, 20, 40 judgments and lr 0.003 with 100 steps for 60 (the optimum sits at the grid's corner, and a 5-replicate check not used for selection prefers 0.001 at 60 as well). The old schedule is clearly worse at 10–40 judgments (validation log-loss 0.75 against 0.51 at 10, 0.55 against 0.31 at 40). Rerunning the whole comparison (same 35 seeds, 32 variants, Split B reusing Split A's schedule) changes the levels and some conclusions:

- Random's log-loss falls by 0.03–0.14 at every budget (e.g. Split A single-shot at 10 judgments 0.666 to 0.528), while its AUC (−0.005 to +0.011) and accuracy (about ±0.015) hardly change, which suggests that the old schedule mostly produced over-confident logits.
- No variant is better than Random on log-loss after Holm in any of the four tables. The two positive primary-metric results above shrink: the all-head Cluster-quota rule in Split B sequential from +0.074 to +0.021 (Holm 1.0), and the AUC gain of deep-ensemble BALD × P(decisive) in Split B sequential from +0.014 to +0.006 (Holm 1.0); DPP in Split B single-shot goes from −0.104 to −0.040 (Holm 0.053). These were partly schedule effects.
- Uncertainty-driven rules look clearly worse than Random in Split B single-shot once Random is well trained (five variants Holm-significantly worse on log-loss, e.g. Uncertainty −0.043, all-head Uncertainty −0.045, Image-coverage uncertainty −0.051; seven on AUC).
- The accuracy gain of deep-ensemble BALD × P(decisive) stays significant under both schedules in three of four cells (Split A single-shot +0.015 → +0.019, Split A sequential +0.020 → +0.017, Split B sequential +0.022 → +0.016), while Fisher D-optimal and ensemble BALD stay positive but lose significance in several cells under the new schedule (e.g. Fisher Split A sequential +0.024 → +0.016, Holm 0.12). BALD × P(decisive) is positive in all 14 table-by-metric cells of the new run (Holm-significant on calibrated log-loss in Split A, +0.022 single-shot and +0.057 sequential, and on AUC in Split B sequential, +0.017), but this was found after the fact, as one rule out of 32 without correction across tables, and is a hypothesis for a pre-registered follow-up.
- The Spearman correlation of the 32 variants' gains between the two schedules is 0.58 for log-loss, 0.55 for calibrated log-loss, 0.80 for AUC and 0.84 for accuracy (pooled over the four tables), so which rule looks best on log-loss is moderately schedule dependent while the AUC and accuracy orderings are stable.

The sensitivity run with a random initial set was not repeated under the new schedule, and the schedule was calibrated on three seeds, one batch per seed and budget.

**Sensitivity to the initial set.** With 10 random judgments as the initial set (20 seeds, both splits, both conditions) no variant is significant on any metric, and the rankings of the variants' log-loss gains agree only weakly with the main run (Spearman −0.01 to 0.63; AUC 0.47–0.74), so the per-rule details depend on the initial set as well as on the seeds. This run has lower power.

**Limitations.** Test sets are small (about 63 and 54 decisive judgments); per-seed gains have a standard deviation of 0.08–0.18 in log-loss; Holm is applied within one table, not across metrics, conditions, splits or budgets; the head schedule was calibrated on the group-unit design and not retuned for 10–60 judgments; coverage is measured in pair space, so a sibling judgment of an already-labelled pair counts as covered; and the pair-level rules cannot concentrate budget on one pair whereas the row-level rules can. The type of a candidate is the queried type and is known to the strategy, the outcome is not. One encoder (frozen SimCLR) and one head schedule were used.

**What §5.14 changes.** The conclusion that no strategy is robustly better than random selection holds under the query unit that matches the labelling software, and the evidence for individual rules is weaker than in §5.12–5.13. It does not support the suspicion that counting the budget in groups had hidden a strategy effect, but it does not exclude one on a larger pool, with another encoder, or with a different head schedule.

### 5.15 Pool-wide variance-reduction design: pre-registered confirmation on new seeds

**Motivation.** §5.14 left open whether any rule beats Random once the head is trained with a schedule suited to 10–60 judgments, and whether the earlier "winners" were noise. This section reports the overnight study (`NEW_METHODS_RESULTS.md`, `NEW_METHODS_LOG.md`, five pre-registration files): development on seeds 600–629, then five confirmatory runs on fresh seeds, each pre-registered in a committed file before the cells existed. All use the protocol of §5.14 (one (pair, type) judgment per query; budgets 10/20/40/60 judgments; single-shot and sequential; Splits A and B; frozen encoder; the **re-tuned per-budget head schedule**) and the same endpoint; the seeds are new random splits of the same 168 pair groups, so each run tests sensitivity to the split draw and not new data.

**Methods.** `vopt_u` is a greedy I-optimal design on the last layer of the own-type head: with the Laplace posterior covariance Σ of the head's last layer (revealed judgments inform only their own head), the gain of candidate *i* is $w_i/(1+w_i\phi_i^\top\Sigma\phi_i)\sum_j(\phi_j^\top\Sigma\phi_i)^2$, the decrease of the summed logit variance over the candidate pool (rank-one updates make the batch aware of its own redundancy). `vopt_u_inf1` multiplies the gain by the predicted probability that the judgment is decisive. For comparison, Fisher D-optimal design (`fisher_dopt`, Active Reward Modeling) and BALD × P(decisive) (`bald_decisive`) from §5.12 were run on the same seeds. `vopt_htr` restricts the same design to HTR candidates, `vopt_hw4` multiplies the gain of HTR candidates by 4, and `random_htr` (random among HTR judgments) separates label allocation from design.

**Main protocol.** On seeds 700–734 (pre-registration 1; 6 tests, Holm 6) the two methods improved decisive accuracy by +0.0095 and +0.0114 (Holm 0.0004, 0.0002; 77% of seeds better) and AUC by +0.0058 and +0.0049 (nominal p 0.03, 0.06; Holm 0.13, 0.18), log-loss not significantly (+0.007, +0.005); development had suggested larger effects (AUC +0.009; winner's curse). On seeds 800–834 (pre-registration 2, replication of the family, AUC and accuracy for four methods, Holm 8) **all eight tests are significant**:

| method | AUC gain (Holm) | accuracy gain (Holm) | log-loss gain (secondary) |
|---|---:|---:|---:|
| `vopt_u` | +0.0127 (0.0004) | +0.0167 (<0.0001) | +0.035 |
| `vopt_u_inf1` | +0.0116 (0.0013) | +0.0156 (<0.0001) | +0.025 |
| `fisher_dopt` | +0.0071 (0.016) | +0.0160 (0.0001) | +0.002 (n.s.) |
| `bald_decisive` | +0.0098 (0.0013) | +0.0153 (<0.0001) | +0.022 |

Pooling both sets (70 seeds, post hoc): `vopt_u` AUC +0.0093, accuracy +0.0131, log-loss +0.0211. In exploratory paired comparisons (uncorrected) `vopt_u` is ahead of Fisher D-optimal design on log-loss and AUC (seeds 800–834: +0.033, p 0.001; +0.0056, p 0.03) and, with the gentle schedule below, of both Fisher and BALD × P(decisive) (AUC +0.0064 and +0.0063, p 0.007 and 0.002), but not in the cold start or against BALD × P(decisive) with the re-tuned schedule; the claim is therefore about the family of last-layer design rules, whereas plain uncertainty sampling (AUC −0.004, log-loss −0.017 on seeds 700–734) and relation-aware core-set (AUC +0.0005, accuracy +0.002) do not share it. Gains are largest at the smallest budgets (`vopt_u`, seeds 800–834: accuracy +0.014/+0.019/+0.018/+0.016 and AUC +0.014/+0.016/+0.012/+0.008 at 10/20/40/60 judgments).

**Cold start.** With 10 random judgments as the initial set instead of the 10 initial groups (about 30 judgments; pre-registration 3, seeds 900–934, 12 tests, Holm 12) every test is significant (all Holm < 0.001): `vopt_u_inf1` log-loss +0.061, AUC +0.020, accuracy +0.024 (86–91% of seeds better); `vopt_u` +0.055/+0.018/+0.019; Fisher +0.038/+0.017/+0.021; BALD × P(decisive) +0.056/+0.019/+0.019. This protocol was first run post hoc as a robustness check on seeds 700–719, which is why it was repeated on fresh seeds.

**Cross-world replication.** The 168 pair groups fall into 117 connected components of the shared-image relation and can be split into two image-disjoint halves. In pre-registration 6 (seeds 1200–1234, 12 tests, Holm 12) the rules were applied in one half (initial set and pool) and evaluated on all groups of the other half (129–135 decisive judgments), in both directions: `vopt_u` improved log-loss by +0.017, AUC by +0.0067 and accuracy by +0.013 (Holm 0.003, 0.002, <0.0001), `vopt_u_inf1` by +0.021/+0.009/+0.017 (all <0.001), BALD × P(decisive) by +0.019/+0.006/+0.015 (all <0.01) and Fisher D-optimal by +0.005 (n.s.)/+0.0046/+0.0075 (0.034, 0.005). The two directions differ: gains are small for H1→H2 (`vopt_u` AUC +0.002, p 0.12; Fisher and BALD AUC slightly negative) and large for H2→H1 (AUC +0.011 to +0.017), so the effect holds on image-disjoint data but depends on the world; `vopt_u_inf1` is positive in both.

**Dependence on the number of labels already in hand.** The gain decays quickly with the size of the initial labelled set (development seeds, Split B, `vopt_u` accuracy gain over Random): +0.019 from a cold start of 10 random judgments (pre-registered, seeds 900–934), +0.017 with the 10 initial groups (about 30 judgments), +0.007 with 20 groups, +0.001 with 40 groups and −0.003 with 60 groups (about 175 judgments; AUC +0.018, +0.007, +0.0003, +0.0002, −0.0013). With 60 initial groups none of the rules helps (uncertainty, BALD × P(decisive), Laplace BALD, gap, core-set, uncertainty + diversity, Cluster-Margin: all within about ±0.007 accuracy), and the HTR priority weight gives no HTR gain (+0.001 accuracy). With the gentle schedule (lr 0.0003, 15 seeds) the gain is also absent at 30 initial groups (about 90 judgments: AUC +0.0016, accuracy +0.0015) and at 60 groups (AUC −0.0003, accuracy +0.0006). In this data set acquisition design is a small-label-set effect.

**Type-targeted design (HTR).** Per-type held-out metrics were added (about 13–16 decisive HTR test judgments per seed). On seeds 1000–1034 (pre-registration 4, 12 tests, all significant after Holm) `vopt_htr` improved HTR accuracy by +0.071, HTR AUC by +0.078 (33 seeds with a defined AUC in all cells) and HTR log-loss by +0.250 over Random, and by +0.021, +0.011 and +0.043 over `random_htr`, so about two thirds of the HTR gain is label allocation and one third is design; the other types pay (accuracy c6x2 −0.030, 1×1 −0.015, √13 −0.005; all-type accuracy +0.002). A soft priority weight (`vopt_hw4`, HTR gain × 4; pre-registration 5, seeds 1100–1134, 5 tests, all Holm < 0.0001) gives HTR accuracy +0.067, HTR AUC +0.058, HTR log-loss +0.183 and all-type accuracy +0.016 and AUC +0.014, with no price elsewhere (accuracy √13 +0.009, c6x2 +0.009, 1×1 −0.002); relative to un-prioritised `vopt_u` it adds HTR accuracy +0.019, AUC +0.014 and log-loss +0.054 and leaves the all-type metrics unchanged.

**Dependence on the training schedule.** With the old schedule (lr 0.01, 100 steps) the gains of the pre-registered methods vanish (`vopt_u` AUC +0.0007, accuracy +0.004; `vopt_u_inf1` AUC +0.0055, accuracy +0.0082, n.s.; 20 seeds), consistent with §5.14. They are present when the head is trained lightly: on seeds 700–714 (descriptive) the AUC gain of `vopt_u` is +0.0025 at lr × steps = 1.0 (old), +0.011 at 0.3 (lr 0.003 × 100 steps or lr 0.001 × 300), +0.008 under the re-tuned schedule (about 0.1) and +0.017 at 0.03 (lr 0.0003 × 100); on development seeds 600–609 the gain is about the same across lr × steps 0.01–0.1 (AUC +0.006 to +0.013, accuracy +0.010 to +0.016), so a further increase for gentler heads is not established and is confounded with seed-set variation (the gain of the same rule is +0.006 on seeds 700–734, +0.013 on 800–834 and +0.021 on 1300–1334 in AUC). The re-tuned schedule sat at the corner of its calibration grid, so the gentlest setting was run as a pre-registered sensitivity analysis (pre-registration 7, seeds 1300–1334, main protocol, 12 tests, Holm 12): **all 12 tests are significant (all p < 0.0001)**, `vopt_u` log-loss +0.044, AUC +0.021, accuracy +0.021 (89%, 91%, 94% of seeds better), `vopt_u_inf1` +0.038/+0.019/+0.021, Fisher +0.032/+0.015/+0.017, BALD × P(decisive) +0.028/+0.015/+0.015. The gentle schedule does not weaken the baseline (Random log-loss 0.423, AUC 0.900, accuracy 0.829 on seeds 1300–1334, against 0.448, 0.899, 0.828 under the re-tuned schedule on seeds 700–714), and `vopt_u` reaches 0.378, 0.921 and 0.850, the best absolute levels of the study. Whether to recalibrate the schedule over a wider grid is a protocol decision left open.

**What did not matter or work.** Ridge (0.3–10), sensitivity-weighted targets, A-optimality, type quotas, a predicted-decisiveness factor, and the neural-tangent-kernel version of the design (all head parameters) changed the gain by less than the seed noise. Random selection among truly decisive judgments (perfect knowledge) is no better than Random, i.e. ties and not-applicable judgments are as useful as decisive ones for this endpoint; the picks of the design rules are not more decisive than random ones (49–52% against 51–52%). A hindsight oracle that uses the test labels beats Random by 0.16–0.20 log-loss and 0.04–0.05 accuracy, so the headroom is large, but the true utility of a single judgment is uncorrelated with every observable feature tried (Spearman |ρ| ≤ 0.06 on 8 seeds), so no per-candidate score can reach it.

**Limitations.** (i) Same 168 groups in every seed: the effect is established under resampling of the splits and not on new data; Wilcoxon p-values are optimistic because seeds resample the same groups. (ii) The first confirmation was weaker than development; main-protocol effects are modest (accuracy +1 to +2 points, AUC +0.006 to +0.013). (iii) HTR test sets are small, so HTR numbers rest on 35-seed averages. (iv) The effect requires a head schedule suited to small label sets. (v) Accuracy was not a primary endpoint of §5.12–5.14 but is the pairwise metric of classifier2. Full tables with per-cell gains: `results/new_methods/CONFIRMATORY_TABLES.md`.

### 5.16 Graph-aware acquisition on an image-similarity graph

**Question.** Pair groups share images and expert comparisons implicitly order the images, so a graph over images might carry information that the mean-pair embedding of the earlier strategies discards. We asked whether a graph-aware selection rule improves held-out preference prediction over Random and over the embedding-based rules.

**What the comparison graph looks like.** Taking images as nodes and the 168 usable pair groups as edges, the graph has 284 nodes and 117 connected components (largest 11 nodes); 237 images occur in exactly one pair group, 42 in two and 5 in three. Per reconstruction type the directed graph of decisive judgments (loser to winner) has 45–94 edges, connected components of at most 4–5 nodes, and only 0–6 images that both won and lost a comparison. The comparison graph is therefore nearly a matching, and a model that recovers a global ranking from it (such as GNNRank, He et al., ICML 2022) has no transitive structure to recover. (The working slides give 669 pairs and about 300 images; the data in this repository contain 638 judgment rows, 168 pair groups and 284 images.) The graph used below is instead built from image similarity.

**Graph and strategies.** At every selection step the graph has as nodes the images of the candidate judgments and of the judgments already revealed, and as edges the symmetrised ten nearest neighbours in the cached SimCLR feature space. It uses no labels and cannot contain a validation or test image, because a selector sees only these images. Three rules were implemented in `graph_strategies.py`: (i) own-head uncertainty multiplied by the percentile rank of the PageRank of the pair's two images, (ii) own-head uncertainty multiplied by a boundary score (the share of an image's neighbours that fall in another k-means cluster of eight), and (iii) farthest-first k-centre on [(a+b)/2, |a−b|, a·b] of features propagated twice over the graph (SGC). Each has a control in which the graph scores, or the propagated features, are permuted over the nodes. Reconstruction-type information (the ideal images as typed anchors) is not available to the selectors in this implementation, so the boundary score is an approximation of the cross-type boundary seen in the ideal images (112 of 770 five-nearest-neighbour edges among ideal images join different types, 52 of them HTR–RT13).

**Protocol.** The query unit is one (pair, type) judgment and the budget counts judgments (10, 20, 40, 60), as in the judgment-unit study; numbers are therefore not comparable with the group-budget tables of §5.12–5.13. Splits A and B, seeds, head schedule and endpoint are those of §5.12–5.13 (35 seeds, single-shot and sequential rounds of 10). The candidate pool holds 201–250 judgments in split A and 311–350 in split B. Gains are per-seed differences from the mean of five Random draws, averaged over budgets; the test is a Wilcoxon signed-rank test over seeds with Holm correction over the seven non-random rows of each block. There are eight blocks (two splits, two conditions, two metrics) and no correction across them.

**Diagnostic before the experiment.** As a check on whether graph smoothing changes what the features can predict, a linear Bradley–Terry head on frozen features was fit to the 264 decisive judgments (five folds by pair group, 20 repetitions). Raw features reach AUC 0.941; features smoothed over the similarity graph (1–4 steps, with or without a within-session temporal chain) reach 0.938–0.940, and a spectral embedding 0.909. This does not test a trained graph network or an acquisition rule.

| Split | Condition | Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |
|---|---|---|---:|---:|---:|---:|---:|
| A | single-shot | Uncertainty x PageRank of the pair's images | +0.034 | 0.106 | +0.006 | 0.588 | 77% (35) |
| A | single-shot | Core-set on propagated features, graph shuffled | +0.028 | 1.000 | +0.003 | 1.000 | 66% (35) |
| A | single-shot | Uncertainty (no graph) | +0.012 | 1.000 | +0.002 | 1.000 | 54% (35) |
| A | single-shot | Uncertainty x boundary score | +0.009 | 1.000 | -0.003 | 1.000 | 57% (35) |
| A | single-shot | Uncertainty x boundary score, graph shuffled | +0.007 | 1.000 | +0.004 | 1.000 | 51% (35) |
| A | single-shot | Core-set on graph-propagated features | +0.003 | 1.000 | -0.005 | 1.000 | 46% (35) |
| A | single-shot | Uncertainty x PageRank, graph shuffled | -0.030 | 1.000 | +0.000 | 1.000 | 46% (35) |
| A | sequential | Uncertainty x PageRank of the pair's images | +0.048 | 0.083 | +0.011 | 0.087 | 66% (35) |
| A | sequential | Core-set on graph-propagated features | +0.026 | 1.000 | +0.000 | 1.000 | 57% (35) |
| A | sequential | Core-set on propagated features, graph shuffled | +0.025 | 1.000 | +0.001 | 1.000 | 54% (35) |
| A | sequential | Uncertainty x boundary score | +0.002 | 1.000 | -0.002 | 1.000 | 60% (35) |
| A | sequential | Uncertainty (no graph) | -0.007 | 1.000 | +0.004 | 1.000 | 46% (35) |
| A | sequential | Uncertainty x boundary score, graph shuffled | -0.013 | 1.000 | +0.004 | 1.000 | 46% (35) |
| A | sequential | Uncertainty x PageRank, graph shuffled | -0.033 | 1.000 | +0.000 | 1.000 | 37% (35) |
| B | single-shot | Core-set on propagated features, graph shuffled | +0.012 | 1.000 | -0.001 | 0.883 | 51% (35) |
| B | single-shot | Core-set on graph-propagated features | +0.004 | 1.000 | -0.003 | 0.883 | 57% (35) |
| B | single-shot | Uncertainty x PageRank of the pair's images | -0.021 | 0.505 | -0.005 | 0.475 | 43% (35) |
| B | single-shot | Uncertainty x PageRank, graph shuffled | -0.045 | 0.018 | -0.014 | 0.019 | 29% (35) |
| B | single-shot | Uncertainty (no graph) | -0.047 | 0.055 | -0.013 | 0.065 | 31% (35) |
| B | single-shot | Uncertainty x boundary score, graph shuffled | -0.053 | 0.025 | -0.008 | 0.280 | 34% (35) |
| B | single-shot | Uncertainty x boundary score | -0.066 | 0.007 | -0.017 | 0.004 | 26% (35) |
| B | sequential | Core-set on propagated features, graph shuffled | +0.056 | 0.013 | +0.005 | 0.803 | 71% (35) |
| B | sequential | Core-set on graph-propagated features | +0.046 | 0.188 | +0.005 | 0.894 | 60% (35) |
| B | sequential | Uncertainty x PageRank, graph shuffled | +0.033 | 0.614 | +0.003 | 1.000 | 57% (35) |
| B | sequential | Uncertainty x PageRank of the pair's images | +0.032 | 0.228 | +0.005 | 0.894 | 74% (35) |
| B | sequential | Uncertainty (no graph) | +0.024 | 0.715 | +0.003 | 1.000 | 57% (35) |
| B | sequential | Uncertainty x boundary score, graph shuffled | -0.004 | 1.000 | -0.002 | 1.000 | 54% (35) |
| B | sequential | Uncertainty x boundary score | -0.017 | 1.000 | -0.009 | 0.812 | 49% (35) |

**Results.** No graph rule is significantly better than Random after Holm correction in any block. The one significant gain in the table belongs to a control: the shuffled-graph core-set in split B sequential (log-loss +0.056, Holm p = 0.013), which shows that the coverage gain of that rule does not come from the graph. Uncertainty multiplied by PageRank is the only rule that is nominally positive in split A in both conditions (log-loss +0.034 single-shot and +0.048 sequential, raw p of about 0.012–0.015 but Holm p of 0.11 and 0.08; AUC +0.006 and +0.011); it is not positive in split B single-shot (−0.021) and its sequential gain there (+0.032, Holm p = 0.23) is matched by its shuffled control (+0.033). The boundary-score rule is never better than Random and is significantly worse in split B single-shot (log-loss −0.066, Holm p = 0.007; AUC −0.017, Holm p = 0.004); uncertainty without a graph is also below Random there (−0.047, Holm p = 0.055), so the boundary score does not repair the weakness of uncertainty sampling in that split. Core-set on propagated features does not differ from its shuffled control in either split.

Comparison with the shuffled controls (paired over seeds, unadjusted p-values, twelve comparisons per metric):

| Split | Condition | Graph rule | log-loss: real − shuffled | p | AUC: real − shuffled | p |
|---|---|---|---:|---:|---:|---:|
| A | single-shot | Uncertainty x PageRank of the pair's images | +0.064 | 0.037 | +0.0058 | 0.081 |
| A | single-shot | Uncertainty x boundary score | +0.002 | 0.865 | -0.0072 | 0.168 |
| A | single-shot | Core-set on graph-propagated features | -0.026 | 0.404 | -0.0077 | 0.225 |
| A | sequential | Uncertainty x PageRank of the pair's images | +0.081 | <0.001 | +0.0111 | 0.031 |
| A | sequential | Uncertainty x boundary score | +0.016 | 0.752 | -0.0056 | 0.287 |
| A | sequential | Core-set on graph-propagated features | +0.001 | 0.929 | -0.0002 | 0.968 |
| B | single-shot | Uncertainty x PageRank of the pair's images | +0.024 | 0.168 | +0.0088 | 0.135 |
| B | single-shot | Uncertainty x boundary score | -0.013 | 0.512 | -0.0095 | 0.073 |
| B | single-shot | Core-set on graph-propagated features | -0.008 | 0.554 | -0.0016 | 0.884 |
| B | sequential | Uncertainty x PageRank of the pair's images | -0.001 | 0.840 | +0.0016 | 0.617 |
| B | sequential | Uncertainty x boundary score | -0.013 | 0.692 | -0.0069 | 0.310 |
| B | sequential | Core-set on graph-propagated features | -0.010 | 0.252 | -0.0001 | 0.777 |

The PageRank rule is better than its shuffled control in split A (log-loss +0.064 single-shot, p = 0.037; +0.081 sequential, p < 0.001), but the shuffled control is itself worse than Random there (−0.030 and −0.033), so part of this difference reflects the control rather than the graph, and the difference is absent in split B.

**What this does and does not show.** Under these splits and with this encoder, selection on a ten-nearest-neighbour similarity graph is not better than Random or than the rules of §5.12–5.13, and the one nominally positive rule does not replicate across splits. The experiment does not test a trained graph network, a graph built from all 1,124 trajectory images (840 of them unlabelled), ideal images as typed anchors, or temporal edges within a session (the temporal constraints file marks these as tentative); the effect of an informative graph could be larger than that of these simple versions. The tests are low-powered for effects of the size seen elsewhere in this report (log-loss gains of 0.03–0.05 against a per-seed standard deviation of 0.09–0.14).

### 5.17 Type-aware graph selection rules: development result and confirmation

**Question.** After §5.16 (first-generation graph rules, nothing significant) we built rules that use the typed reference images: label propagation of the reference-image type posterior over a kNN graph (k = 10, alpha = 0.9) gives compact graph features per judgment, from which P(decisive) is fitted on the revealed judgments; the rules combine farthest-first coverage in a propagated-feature space with P(decisive) (and, for `coverage_unc`, the head's own uncertainty; for `bald`, BALD). Selectors see the typed reference images, which are already training anchors of every strategy; this is an open protocol decision for the advisor.

**Procedure (fixed before running; this section was produced by two independent runs of the same frozen protocol that gave identical numbers, one analysed with Holm over 3 candidates within each block, the other with Holm over all 12 candidate tests as shown here).** Design and tuning on DEV seeds 400-409 only; three candidates frozen (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`; controls `..._unc_typeonly` = decisive predictor from the type one-hot only, `..._unc_shuffled` = shuffled posterior); then one run on 25 confirmation seeds (410-429, 42, 79, 123, 202, 303). Holm over 3 candidates x 4 blocks = 12 tests per metric. Success: log-loss Holm p < 0.05 in at least two of the four blocks including both splits and a non-negative AUC gain. All numbers below come from `aggregate_confirm.py` on the stored cells.

**Development result (10 seeds, not evidence).** `coverage_unc` showed log-loss gains of +0.108 / +0.138 / +0.029 / +0.091 in A-single / A-sequential / B-single / B-sequential, and the type-only control already gave most of it (+0.119 / +0.056 / -0.017 / +0.056).

**Confirmation result.**

#### log-loss: gain over Random, 25 confirmation seeds (mean [95% bootstrap CI], share of seeds better, raw Wilcoxon p, Holm p over 12 candidate tests)

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage_unc | -0.002 [-0.050, +0.044] (48%, p 0.979, Holm 1.000) | -0.008 [-0.062, +0.040] (60%, p 0.692, Holm 1.000) | +0.002 [-0.035, +0.041] (48%, p 0.895, Holm 1.000) | +0.050 [+0.004, +0.095] (72%, p 0.027, Holm 0.275) |
| typed_decisive_coverage | -0.007 [-0.052, +0.038] (56%, p 0.895, Holm 1.000) | -0.010 [-0.052, +0.030] (48%, p 0.812, Holm 1.000) | +0.062 [+0.022, +0.101] (76%, p 0.005, Holm 0.051) | +0.083 [+0.041, +0.125] (80%, p 0.001, Holm 0.008) |
| typed_decisive_bald | +0.007 [-0.046, +0.061] (60%, p 0.979, Holm 1.000) | -0.006 [-0.069, +0.056] (56%, p 0.916, Holm 1.000) | +0.019 [-0.024, +0.059] (60%, p 0.312, Holm 1.000) | +0.049 [+0.004, +0.095] (76%, p 0.042, Holm 0.379) |
| typed_decisive_coverage_unc_typeonly (control) | +0.026 [-0.012, +0.065] (60%, p 0.191) | +0.030 [-0.013, +0.068] (64%, p 0.067) | -0.005 [-0.035, +0.024] (44%, p 0.672) | +0.040 [-0.009, +0.084] (72%, p 0.127) |
| typed_decisive_coverage_unc_shuffled (control) | +0.040 [-0.000, +0.082] (64%, p 0.059) | +0.006 [-0.026, +0.036] (52%, p 0.653) | +0.031 [+0.002, +0.062] (64%, p 0.067) | +0.081 [+0.048, +0.114] (84%, p 0.000) |

n seeds per cell: [25]

#### AUC: gain over Random, 25 confirmation seeds (mean [95% bootstrap CI], share of seeds better, raw Wilcoxon p, Holm p over 12 candidate tests)

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage_unc | +0.007 [-0.001, +0.015] (64%, p 0.156, Holm 1.000) | +0.006 [-0.003, +0.015] (56%, p 0.411, Holm 1.000) | +0.005 [-0.002, +0.013] (52%, p 0.381, Holm 1.000) | +0.015 [+0.004, +0.024] (72%, p 0.005, Holm 0.046) |
| typed_decisive_coverage | -0.002 [-0.010, +0.005] (44%, p 0.578, Holm 1.000) | -0.001 [-0.010, +0.009] (56%, p 0.874, Holm 1.000) | +0.013 [+0.006, +0.020] (72%, p 0.002, Holm 0.025) | +0.018 [+0.011, +0.026] (84%, p 0.000, Holm 0.001) |
| typed_decisive_bald | +0.006 [-0.003, +0.015] (64%, p 0.230, Holm 1.000) | +0.006 [-0.006, +0.017] (56%, p 0.396, Holm 1.000) | +0.006 [-0.003, +0.015] (64%, p 0.210, Holm 1.000) | +0.015 [+0.005, +0.025] (72%, p 0.009, Holm 0.079) |
| typed_decisive_coverage_unc_typeonly (control) | -0.000 [-0.013, +0.011] (60%, p 0.653) | +0.007 [-0.001, +0.015] (64%, p 0.134) | -0.003 [-0.010, +0.004] (44%, p 0.596) | +0.004 [-0.007, +0.013] (60%, p 0.287) |
| typed_decisive_coverage_unc_shuffled (control) | +0.003 [-0.008, +0.013] (64%, p 0.442) | +0.004 [-0.004, +0.012] (56%, p 0.275) | +0.003 [-0.003, +0.009] (60%, p 0.442) | +0.015 [+0.008, +0.021] (80%, p 0.001) |

n seeds per cell: [25]

#### Real minus control, log-loss gain (paired over seeds; positive = the graph / uncertainty signal adds over the control)

| comparison | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| coverage_unc minus typeonly | -0.028 [-0.079, +0.025] (p 0.339) | -0.038 [-0.096, +0.017] (p 0.353) | +0.007 [-0.033, +0.047] (p 0.751) | +0.011 [-0.033, +0.060] (p 0.812) |
| coverage_unc minus shuffled | -0.042 [-0.097, +0.006] (p 0.442) | -0.013 [-0.065, +0.036] (p 0.833) | -0.029 [-0.055, -0.003] (p 0.067) | -0.031 [-0.065, +0.005] (p 0.075) |

#### Success criterion (log-loss Holm p < 0.05 in >= 2 of 4 blocks incl. both splits; mean AUC gain >= 0 in those blocks)

- typed_decisive_coverage_unc: significant blocks [] -> NOT met
- typed_decisive_coverage: significant blocks ['B-sequential'] -> NOT met
- typed_decisive_bald: significant blocks [] -> NOT met

**What this shows.**
- The success criterion is **not met** by any candidate. The development gains in split A did not replicate: with 25 fresh seeds `coverage_unc` is at -0.002 and -0.008 in A, so the +0.11 / +0.14 seen on 10 seeds was most likely selection on noise (winner's curse), not an effect.
- The only block that passes Holm is B-sequential for plain `typed_decisive_coverage` (+0.083 log-loss, Holm p 0.008; AUC +0.018). B-single reaches Holm p 0.051 (just above the line). Both are in split B (20% pair-group hold-out) only.
- That effect is not clearly due to the graph: the shuffled-posterior control of `coverage_unc` reaches +0.081 in B-sequential (p < 0.001), and `coverage_unc` minus its shuffled control is negative in B. So what helps there is the coverage structure and the type-dependent decisive prior, not the label-propagated type posterior. We have not run the shuffled control for plain `coverage`.
- Gains are small relative to per-seed variability (sd roughly 0.1 log-loss), and AUC gains are at most +0.018.

**Limitations.** Only three candidates and two controls were confirmed; variants tried on DEV (g2, u2, q-coordinates, decisive-weighted sampling, GCN baselines) were not. The shuffled/typeonly controls were run only for `coverage_unc`. Seeds 42, 79, 123, 202, 303 were also used in §5.16, so they are not independent of the first-generation rules, but they were not used to tune the new rules.

### 5.18 Replication test of the type-aware coverage rule on fresh seeds

**Why.** In §5.17 only plain `typed_decisive_coverage` passed Holm (B-sequential), and its shuffled-graph control was just as good for a related rule, so we could not tell whether the effect was real or caused by the graph. Before running, we wrote down (AUTONOMOUS_RUN_LOG.md): one candidate, `typed_decisive_coverage`; replication = Holm p < 0.05 and positive log-loss gain in both B-single and B-sequential with non-negative AUC gain; seeds 430-449, never used before; its shuffled-posterior and type-only controls and two non-graph coverage baselines (`core_set_relation`, `typiclust_pairs`) run on the same seeds. A graph claim additionally requires beating both controls.

**Literature check that preceded it.** Low-budget active learning is reported to favour coverage/typicality over uncertainty (TypiClust, Hacohen et al. 2022; ProbCover, Yehuda et al. 2022), and an oracle that cannot answer some queries is handled by weighting selection with the probability that the oracle answers (Du and Ling, "Active learning from oracle with knowledge blind spot", AAAI 2013). Our P(decisive) weighting is of the second kind; we read these only through search summaries, not the full papers. Because our farthest-first coverage prefers outliers, we also tried a ProbCover-style weighted maximum-cover rule (radius = 2%, 5%, 10% quantile of pairwise distances). On development seeds 400-409 it was not better than farthest-first (log-loss gain A-single/A-seq/B-single/B-seq: q02 -0.009/+0.059/+0.015/+0.065, q05 -0.037/+0.012/-0.004/+0.097, q10 -0.002/+0.041/-0.002/+0.071, against +0.095/+0.109/+0.024/+0.105 for farthest-first) and was dropped without confirmation.

**Result (20 fresh seeds).**

#### log-loss: gain over Random, seeds 430-449 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over the 4 blocks for the candidate))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage | +0.039 [-0.004, +0.077] (65%, p 0.064, Holm 0.255) | +0.019 [-0.016, +0.054] (70%, p 0.216, Holm 0.432) | +0.041 [+0.004, +0.081] (65%, p 0.097, Holm 0.292) | +0.042 [-0.012, +0.102] (55%, p 0.330, Holm 0.432) |
| typed_decisive_coverage_typeonly (control/reference) | +0.033 [-0.016, +0.073] (75%, p 0.083) | +0.046 [+0.004, +0.094] (60%, p 0.058) | +0.042 [-0.004, +0.088] (70%, p 0.105) | +0.030 [-0.027, +0.086] (55%, p 0.368) |
| typed_decisive_coverage_shuffled (control/reference) | +0.048 [+0.022, +0.073] (85%, p 0.002) | +0.062 [+0.013, +0.118] (70%, p 0.015) | +0.062 [+0.023, +0.103] (80%, p 0.009) | +0.050 [-0.008, +0.109] (65%, p 0.154) |
| core_set_relation (control/reference) | +0.072 [+0.039, +0.106] (80%, p 0.001) | +0.072 [+0.021, +0.129] (80%, p 0.004) | -0.027 [-0.082, +0.026] (45%, p 0.546) | -0.048 [-0.102, +0.006] (35%, p 0.097) |
| typiclust_pairs (control/reference) | -0.010 [-0.091, +0.048] (75%, p 0.114) | +0.016 [-0.025, +0.059] (55%, p 0.388) | -0.064 [-0.118, -0.012] (30%, p 0.070) | -0.052 [-0.129, +0.012] (40%, p 0.349) |

n seeds per cell: [20]

#### AUC: gain over Random, seeds 430-449 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over the 4 blocks for the candidate))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage | +0.006 [-0.003, +0.015] (55%, p 0.261, Holm 1.000) | +0.001 [-0.008, +0.010] (45%, p 0.985, Holm 1.000) | +0.007 [-0.004, +0.019] (60%, p 0.498, Holm 1.000) | +0.011 [-0.002, +0.026] (50%, p 0.277, Holm 1.000) |
| typed_decisive_coverage_typeonly (control/reference) | -0.000 [-0.011, +0.009] (55%, p 0.756) | +0.000 [-0.008, +0.009] (55%, p 0.898) | +0.006 [-0.007, +0.019] (50%, p 0.596) | +0.001 [-0.013, +0.016] (45%, p 0.956) |
| typed_decisive_coverage_shuffled (control/reference) | +0.005 [-0.002, +0.011] (70%, p 0.177) | +0.005 [-0.005, +0.014] (60%, p 0.368) | +0.009 [-0.001, +0.020] (65%, p 0.154) | +0.006 [-0.005, +0.017] (60%, p 0.349) |
| core_set_relation (control/reference) | +0.008 [-0.002, +0.018] (75%, p 0.083) | +0.006 [-0.005, +0.017] (55%, p 0.294) | -0.006 [-0.019, +0.007] (40%, p 0.498) | -0.014 [-0.031, +0.002] (60%, p 0.452) |
| typiclust_pairs (control/reference) | -0.006 [-0.017, +0.005] (45%, p 0.648) | -0.008 [-0.016, +0.000] (45%, p 0.133) | -0.010 [-0.022, +0.001] (30%, p 0.133) | -0.011 [-0.027, +0.004] (45%, p 0.294) |

n seeds per cell: [20]

#### Paired differences, log-loss (positive = coverage rule is better than the comparison)

| coverage minus | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage_typeonly | +0.006 [-0.045, +0.052] (p 0.546) | -0.027 [-0.083, +0.020] (p 0.596) | -0.001 [-0.042, +0.038] (p 0.927) | +0.013 [-0.034, +0.063] (p 0.956) |
| typed_decisive_coverage_shuffled | -0.009 [-0.066, +0.043] (p 0.812) | -0.043 [-0.108, +0.016] (p 0.165) | -0.021 [-0.069, +0.022] (p 0.571) | -0.007 [-0.056, +0.037] (p 0.956) |
| core_set_relation | -0.034 [-0.098, +0.024] (p 0.452) | -0.053 [-0.121, +0.007] (p 0.216) | +0.068 [+0.010, +0.129] (p 0.083) | +0.090 [+0.028, +0.157] (p 0.024) |
| typiclust_pairs | +0.048 [-0.002, +0.108] (p 0.231) | +0.003 [-0.049, +0.053] (p 0.869) | +0.105 [+0.032, +0.178] (p 0.021) | +0.094 [+0.016, +0.180] (p 0.123) |

Replication criterion (Holm p < 0.05, positive log-loss gain and AUC gain >= 0 in both B-single and B-sequential): NOT met

**What this shows.**
- The B-split result of §5.17 did **not** replicate: gains of +0.041 (B-single) and +0.042 (B-sequential) with raw p 0.097 and 0.330, Holm p 0.29 and 0.43. The earlier +0.062 / +0.083 were probably inflated by choosing this rule after seeing its result (regression to the mean).
- There is no evidence for a graph effect: the shuffled-posterior control is as good or better in all four blocks (coverage minus shuffled -0.043 to -0.007, none significant), and the type-only control is equal within noise.
- The three variants (real, type-only, shuffled graph) all give roughly +0.03 to +0.06 log-loss in every block, which is the same type-aware decisive weighting plus coverage. That is suggestive of a small real gain from this common ingredient, but it was not a pre-registered claim, and each single block is not significant after Holm; we do not claim it.
- The non-graph coverage baseline `core_set_relation` is better than the graph rule in split A (+0.072 / +0.072) and worse in split B; `typiclust_pairs` is not better than Random. AUC gains are within +/-0.015 everywhere.

**Extended replication in split B (30 seeds).** In a separate pre-registered run (`graph_exploration/AUTONOMOUS_RUN_LOG.md`, Round 4) the same plain rule, its two controls and the Core-set baseline were run in split B on seeds 430-459; seeds 430-449 coincide with the run above (the cells are the same), so the two are not independent evidence. Gains over Random, with raw seed-level Wilcoxon p-values (H1 corrected over the two conditions; paired differences are not corrected):

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

In both conditions the gain is about +0.03 to +0.04 log-loss, borderline in single-shot (Holm p = 0.059) and not significant in sequential selection, and the type-only and shuffled-posterior controls give the same gain, so the paired differences are within noise. This agrees with the 20-seed split-B numbers above.

**Overall conclusion of the graph exploration.** Across 4 frozen candidates and 2 fresh confirmation sets (45 seeds in total for the §5.17 candidates, 20 for this test), no graph-aware rule showed a significant, reproducible improvement over Random that could be attributed to the graph. The graph features predict whether a judgment is decisive (AUC 0.83 on all 521 rows, a data-level diagnostic), but that predictor did not translate into better held-out preference prediction at these budgets. Possible reasons, not tested: the head sees only 10-60 judgments so selection effects are small compared with seed variance (sd about 0.1), and the comparison graph is nearly a matching (§5.16), leaving no ranking structure to exploit.

**Caveat on the head schedule and on the role of decisiveness.** All graph rules of §5.16–5.18 were run under the old head schedule (lr 0.01, 100 steps). §5.14 found that this schedule is clearly too aggressive for 10–60 judgments, and §5.15 found that last-layer design rules gain over Random under the re-tuned and gentler schedules but not under the old one; the graph rules were therefore re-tested under those schedules in §5.20, where their conclusion changes (they beat Random). §5.15 also reports that restricting Random to truly decisive judgments is no better than Random and that a predicted-decisiveness factor did not change the gain beyond seed noise, so the decisive weighting used here is not expected to be the active ingredient; what the real, type-only and shuffled variants share, coverage in a propagated feature space combined with a type-dependent weight, is a more plausible source of the small gain seen in split B under the old schedule, and §5.20 measures what the graph-derived type posterior adds to it.

### 5.19 Controls and reference classifiers on the ideal-image endpoint

**Question.** How do simple classifiers compare with the reward-model pipeline when every method uses the same split, the same labelled ideal images (the reference images, about 88 per seed) and the same outer-test images (28 per seed)? These are controls and reference classifiers, not competing acquisition methods: they use the absolute labels of the reference images and no pairwise labels (`active_learning_studies/classification_baselines/`, `README.md`).

**Protocol.** The split is the one of Task 3c (the same `Experiment.load_and_split`, de-duplicated by SHA-256 content identity); hyper-parameters (logistic-regression C, fine-tuning epoch) are selected on the references or on the utility-validation images and never on the outer test. Frozen SimCLR 1-NN reproduces the number already in the repository seed by seed (0.871 ± 0.041; unit test). The five paper seeds (42, 79, 123, 202, 303) are the primary result; 25 further splits (seeds 400-424) are a supplementary check whose test sets overlap, so no significance tests are attached.

| Extractor | Head | Accuracy (5 paper splits, 28 test images each) | Macro-F1 | HTR recall | Δ accuracy vs. frozen SimCLR 1-NN |
|---|---|---:|---:|---:|---:|
| simclr_resnet18 | 1nn | 0.871 ± 0.041 | 0.856 ± 0.046 | 0.76 ± 0.09 | +0.000 |
| simclr_resnet18 | logreg | 0.850 ± 0.047 | 0.827 ± 0.064 | 0.64 ± 0.22 | -0.021 |
| raw_pixels_64 | logreg | 0.886 ± 0.030 | 0.879 ± 0.032 | 0.88 ± 0.18 | +0.014 |
| random_resnet18_init0 | 1nn | 0.900 ± 0.039 | 0.883 ± 0.050 | 0.68 ± 0.18 | +0.029 |
| random_resnet18_init0 | logreg | 0.907 ± 0.048 | 0.898 ± 0.047 | 0.80 ± 0.14 | +0.036 |
| peak_profile_24 | 1nn | 0.871 ± 0.041 | 0.843 ± 0.050 | 0.60 ± 0.20 | +0.000 |
| finetune_simclr_resnet18 | finetune_ce | 0.893 ± 0.051 | 0.880 ± 0.059 | 0.76 ± 0.22 | +0.021 |
| finetune_random_resnet18 | finetune_ce | 0.807 ± 0.070 | 0.777 ± 0.090 | 0.60 ± 0.32 | -0.064 |

(Full table with all extractors and heads: `classification_baselines/results/table_paper_seeds_with_finetune.md`; frozen features on 30 splits: `table_all_30_splits.md`. Not run because the weights could not be downloaded in this environment: ImageNet ResNet-18/50 and ViT-B/16; DINOv2 and CLIP extractors are not written.)

**What it shows.** Every runnable control lands between 0.81 and 0.91 accuracy; the differences between rows are far smaller than the split-to-split sd (0.03-0.09) and than the binomial noise of 28 test images (about ±0.06), so the 28-image test cannot rank them. Pre-training is not visibly needed for this endpoint: random-weight ResNet-18 features and raw pixels with logistic regression match the SimCLR features (over the 30 frozen-feature splits, raw pixels + logistic regression is 0.914 against 0.877 for SimCLR 1-NN, above it in 20 of 27 non-tied splits, a description of these overlapping splits and not a test). All of them are far above the fine-tuned Bradley-Terry reward model of §5.10 (0.4-0.7 on the same seeds' outer-test images) and consistent with the frozen-head result of §5.11 (0.73-0.86), which supports the earlier reading that the low Task 3c accuracy comes from the training procedure and the tiny labelled sets, not from the encoder or the images. HTR is the weakest class for most controls (recall 0.52-0.88, five HTR test images per seed, so one image changes recall by 0.2), and fine-tuning on 88 images is unstable (selected epochs 6-30; the random-weight fine-tune ranges from 0.71 to 0.89 across seeds). The "5-NN with cross-validation over all ideal images" numbers from the slides use a different protocol and are not comparable.

### 5.20 Graph rules under the re-tuned head schedule, and a graph-regularised variance-reduction design

**Why this section.** Every graph experiment of §5.16–5.18 used the head schedule of §5.12–5.13 (lr 0.01, 100 steps). §5.14 showed that this schedule is too aggressive for 10–60 judgments, and §5.15 that the pool-wide variance-reduction rules gain over Random only under the re-tuned schedule (lr 0.001, 100 steps; lr 0.003 at 60 judgments) or a gentler one (lr 0.0003). The graph rules had therefore not been tested in the regime in which other rules of this report show an effect. Everything here uses the judgment-unit protocol of §5.14 (one (pair, type) judgment per query, budgets of 10, 20, 40 and 60 judgments, splits A and B, single-shot and sequential selection, Random = mean of five draws, the held-out preference endpoint) with the per-budget head schedule named in each part; the hypotheses, the seed sets and the freezing of rules were written to `graph_exploration/OVERNIGHT2_PLAN.md` and `AUTONOMOUS_RUN_LOG.md` before the corresponding cells were run. Gains are per-seed differences from Random averaged over the four budgets; p-values are seed-level Wilcoxon signed-rank tests; Holm correction is over the frozen candidates within each block; paired differences to controls carry raw p-values and are not corrected for multiplicity.

**Part 1: the frozen graph rules under the re-tuned schedule (30 fresh seeds, 1500–1529).** The three rules frozen in §5.17 (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`) were run unchanged, with the type-only and shuffled-posterior controls of the plain rule and, as references, the pool-wide I-optimal design without a graph (`vopt_u`, §5.15) and relation-aware Core-set.

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.032 | <0.001 | 0.002 | +0.0105 | +0.0088 | 73% | 30 |
| A | single-shot | ... x own-head uncertainty | +0.022 | 0.002 | 0.002 | +0.0110 | +0.0135 | 80% | 30 |
| A | single-shot | Laplace BALD x P(decisive), graph features | +0.040 | <0.001 | 0.002 | +0.0166 | +0.0157 | 73% | 30 |
| A | single-shot | (control) decisive predictor from the type only | +0.017 | 0.067 |  | -0.0017 | -0.0037 | 63% | 30 |
| A | single-shot | (control) type posterior shuffled over images | +0.027 | 0.040 |  | +0.0033 | +0.0016 | 63% | 30 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.028 | 0.008 |  | +0.0113 | +0.0130 | 70% | 30 |
| A | single-shot | (reference) Core-set, relation-aware pairs | +0.025 | 0.073 |  | +0.0048 | -0.0017 | 67% | 30 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.027 | 0.003 | 0.006 | +0.0079 | +0.0150 | 73% | 30 |
| A | sequential | ... x own-head uncertainty | +0.025 | 0.011 | 0.011 | +0.0105 | +0.0246 | 70% | 30 |
| A | sequential | Laplace BALD x P(decisive), graph features | +0.034 | 0.002 | 0.006 | +0.0127 | +0.0218 | 70% | 30 |
| A | sequential | (control) decisive predictor from the type only | +0.001 | 0.598 |  | -0.0066 | -0.0042 | 57% | 30 |
| A | sequential | (control) type posterior shuffled over images | +0.015 | 0.229 |  | -0.0002 | +0.0056 | 57% | 30 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.021 | 0.119 |  | +0.0100 | +0.0229 | 57% | 30 |
| A | sequential | (reference) Core-set, relation-aware pairs | +0.024 | 0.040 |  | +0.0048 | +0.0136 | 63% | 30 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.047 | <0.001 | 0.003 | +0.0134 | +0.0105 | 73% | 30 |
| B | single-shot | ... x own-head uncertainty | +0.026 | 0.124 | 0.124 | +0.0100 | +0.0200 | 60% | 30 |
| B | single-shot | Laplace BALD x P(decisive), graph features | +0.038 | 0.005 | 0.009 | +0.0149 | +0.0155 | 67% | 30 |
| B | single-shot | (control) decisive predictor from the type only | +0.025 | 0.050 |  | +0.0016 | +0.0037 | 63% | 30 |
| B | single-shot | (control) type posterior shuffled over images | +0.035 | 0.047 |  | +0.0067 | +0.0069 | 67% | 30 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.058 | 0.001 |  | +0.0189 | +0.0150 | 73% | 30 |
| B | single-shot | (reference) Core-set, relation-aware pairs | +0.047 | 0.001 |  | +0.0115 | +0.0060 | 70% | 30 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.050 | 0.003 | 0.009 | +0.0140 | +0.0143 | 70% | 30 |
| B | sequential | ... x own-head uncertainty | +0.032 | 0.070 | 0.140 | +0.0071 | +0.0259 | 70% | 30 |
| B | sequential | Laplace BALD x P(decisive), graph features | +0.034 | 0.084 | 0.140 | +0.0121 | +0.0167 | 60% | 30 |
| B | sequential | (control) decisive predictor from the type only | +0.022 | 0.177 |  | -0.0007 | +0.0063 | 57% | 30 |
| B | sequential | (control) type posterior shuffled over images | +0.028 | 0.088 |  | +0.0025 | +0.0082 | 57% | 30 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.048 | 0.031 |  | +0.0163 | +0.0217 | 63% | 30 |
| B | sequential | (reference) Core-set, relation-aware pairs | +0.047 | 0.025 |  | +0.0088 | +0.0064 | 67% | 30 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0154 | 0.100 | +0.0123 | 0.004 | +0.0125 | 0.023 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0051 | 0.503 | +0.0072 | 0.096 | +0.0072 | 0.198 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0037 | 0.919 | -0.0007 | 0.792 | -0.0042 | 0.265 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0257 | 0.022 | +0.0145 | 0.003 | +0.0192 | 0.003 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0124 | 0.198 | +0.0081 | 0.086 | +0.0094 | 0.068 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0056 | 0.919 | -0.0021 | 0.715 | -0.0079 | 0.096 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0215 | 0.245 | +0.0119 | 0.025 | +0.0068 | 0.239 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0116 | 0.280 | +0.0067 | 0.064 | +0.0037 | 0.370 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0115 | 0.349 | -0.0054 | 0.428 | -0.0045 | 0.781 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0282 | 0.031 | +0.0146 | 0.004 | +0.0080 | 0.133 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0229 | 0.061 | +0.0115 | 0.014 | +0.0060 | 0.198 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0027 | 0.715 | -0.0023 | 0.730 | -0.0074 | 0.238 |

The plain rule has a positive log-loss gain over Random in all four blocks (Holm p between 0.002 and 0.009), with positive AUC and accuracy gains; the Laplace-BALD rule is significant in three blocks and the uncertainty-weighted rule in two. This differs from the old-schedule result of §5.17–5.18 and shows that the earlier null was conditional on the schedule. In contrast with the old schedule, the type posterior is also measurably useful: the plain rule is above its type-only control in AUC in all four blocks (+0.012 to +0.015, raw p between 0.003 and 0.025) and above its shuffled-posterior control in AUC in all four (+0.007 to +0.012, raw p between 0.014 and 0.096, only one below 0.05), and its log-loss differences to the controls are positive but mostly not significant. The rule is not distinguishable from `vopt_u` (log-loss differences −0.012 to +0.006, none significant), so the graph rule is comparable with, not better than, the best non-graph rule of §5.15.

**Part 2: a graph-regularised variance-reduction design (development on seeds 600–624).** The strongest non-graph rule of §5.15, `vopt_u`, is a pool-wide I-optimal design on the last layer of the own-type head under the Laplace posterior. Because the score of a judgment is f(a) − f(b) with f(x) = θ·h(x) and h the hidden layer, a smoothness prior for f over the image graph is a prior on the head parameters θ. `gvopt_lap` adds to the prior precision (ridge · I) the term λ · HᵀLH rescaled to mean diagonal 1, with H the hidden features of the images of the kNN graph (candidates, revealed judgments and typed reference images) and L its Laplacian; this is the Laplacian-regularised optimal design of He (IEEE TIP 2010) and Cai and He (IEEE TKDE 2012) applied to the last layer (`graph_exploration/notes_overnight2_literature.md`). Further variants weighted the pool judgments by the type posterior (`gvopt_type`), designed on propagated features (`gvopt_prop`), or used the Σ-optimal criterion of Ma, Garnett and Schneider (NeurIPS 2013) with and without the graph prior (`gvopt_sigma`, `gvopt_lapsigma`); λ was varied over 0.3, 1, 3 and 10. Development results (mean over the four blocks of the gain over Random; the last three columns are the differences to `vopt_u`):

```
rule             log_loss       auc  accuracy   | minus vopt_u: log_loss       auc  accuracy  blocks all-positive
vopt_u            +0.0133   +0.0076   +0.0164   |                  +0.0000   +0.0000   +0.0000  4/4
gvopt_lap         +0.0135   +0.0095   +0.0186   |                  +0.0002   +0.0019   +0.0022  2/4
gvopt_lap03       +0.0085   +0.0082   +0.0171   |                  -0.0048   +0.0006   +0.0007  2/4
gvopt_lap3        +0.0164   +0.0111   +0.0210   |                  +0.0031   +0.0034   +0.0046  3/4
gvopt_lap10       +0.0122   +0.0115   +0.0204   |                  -0.0011   +0.0039   +0.0040  2/4
gvopt_type        +0.0177   +0.0066   +0.0110   |                  +0.0045   -0.0010   -0.0054  4/4
gvopt_prop        +0.0082   +0.0055   +0.0106   |                  -0.0051   -0.0021   -0.0058  4/4
gvopt_sigma       +0.0029   +0.0031   +0.0062   |                  -0.0103   -0.0045   -0.0102  2/4
gvopt_lapsigma    -0.0009   +0.0045   +0.0100   |                  -0.0142   -0.0031   -0.0064  3/4
```

In split A the Laplacian-prior rules clearly beat Random on the development seeds, in split B nothing beats Random on log-loss (all about 0) and only AUC and accuracy are positive; the Σ-optimal and propagated-feature variants are worse than `vopt_u`. `gvopt_lap3` (λ = 3) was frozen as the graph method because it was the only variant above `vopt_u` on all three metrics; λ was therefore chosen on the development seeds from four values and is a tuned setting.

**Part 3: confirmation of the frozen method under the re-tuned schedule (35 fresh seeds, 1400–1434).**

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) | -0.018 | 0.232 | 0.279 | +0.0082 | +0.0140 | 40% | 35 |
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.014 | 0.140 | 0.279 | +0.0043 | +0.0038 | 57% | 35 |
| A | single-shot | (control) same, graph shuffled | -0.009 | 0.955 |  | +0.0042 | +0.0124 | 60% | 35 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.001 | 0.903 |  | +0.0078 | +0.0164 | 54% | 35 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) | -0.014 | 0.342 | 0.685 | +0.0094 | +0.0202 | 46% | 35 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.008 | 0.471 | 0.685 | +0.0025 | +0.0043 | 54% | 35 |
| A | sequential | (control) same, graph shuffled | +0.003 | 0.441 |  | +0.0070 | +0.0174 | 57% | 35 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.018 | 0.154 |  | +0.0100 | +0.0141 | 54% | 35 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.006 | 0.994 | 0.994 | +0.0092 | +0.0132 | 43% | 35 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.038 | <0.001 | <0.001 | +0.0105 | +0.0099 | 86% | 35 |
| B | single-shot | (control) same, graph shuffled | +0.022 | 0.039 |  | +0.0118 | +0.0113 | 66% | 35 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.025 | 0.023 |  | +0.0101 | +0.0102 | 60% | 35 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.011 | 0.213 | 0.213 | +0.0085 | +0.0160 | 66% | 35 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.028 | 0.004 | 0.008 | +0.0065 | +0.0047 | 69% | 35 |
| B | sequential | (control) same, graph shuffled | +0.012 | 0.219 |  | +0.0058 | +0.0095 | 63% | 35 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.021 | 0.017 |  | +0.0083 | +0.0158 | 66% | 35 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0091 | 0.471 | +0.0040 | 0.287 | +0.0016 | 0.612 |
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0186 | 0.245 | +0.0004 | 0.831 | -0.0024 | 0.602 |
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0315 | 0.040 | +0.0039 | 0.413 | +0.0102 | 0.081 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0177 | 0.385 | +0.0024 | 0.359 | +0.0029 | 0.280 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0320 | 0.046 | -0.0006 | 0.668 | +0.0061 | 0.172 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0224 | 0.207 | +0.0069 | 0.326 | +0.0159 | 0.006 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0156 | 0.075 | -0.0026 | 0.422 | +0.0019 | 0.570 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0190 | 0.154 | -0.0010 | 0.844 | +0.0031 | 0.411 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0320 | 0.014 | -0.0013 | 0.481 | +0.0033 | 0.417 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0012 | 0.840 | +0.0027 | 0.368 | +0.0065 | 0.399 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0107 | 0.815 | +0.0002 | 0.351 | +0.0002 | 0.647 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0173 | 0.441 | +0.0020 | 0.656 | +0.0113 | 0.057 |

The method did not replicate. `gvopt_lap3` has no significant log-loss gain over Random in any block (−0.018, −0.014, +0.006, +0.011), its positive AUC (+0.008 to +0.009) and accuracy (+0.013 to +0.020) gains equal those of its shuffled-graph control and of `vopt_u`, and it is below `vopt_u` on log-loss in split A sequential (−0.032, raw p 0.046). The development advantage in split A was therefore selection among nine variants and four λ values on 25 seeds. The second candidate, the plain typed coverage rule, was significant in split B (Holm < 0.001 and 0.008) and not in split A, so the pre-registered criterion (Holm p < 0.05 in at least three of four blocks) was not met by either rule in this seed set.

**Part 4: the same frozen rules under a gentler schedule (lr 0.0003, seeds 1400–1434).**

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.033 | <0.001 | 0.002 | +0.0179 | +0.0266 | 74% | 35 |
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.014 | 0.046 | 0.046 | +0.0061 | +0.0091 | 63% | 35 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.031 | <0.001 |  | +0.0193 | +0.0271 | 74% | 35 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.039 | <0.001 | <0.001 | +0.0193 | +0.0276 | 74% | 35 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.014 | 0.056 | 0.056 | +0.0040 | +0.0074 | 63% | 35 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.035 | <0.001 |  | +0.0170 | +0.0250 | 80% | 35 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.044 | <0.001 | <0.001 | +0.0182 | +0.0161 | 97% | 35 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.019 | <0.001 | <0.001 | +0.0104 | +0.0103 | 77% | 35 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.033 | <0.001 |  | +0.0148 | +0.0103 | 77% | 35 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.041 | <0.001 | <0.001 | +0.0162 | +0.0196 | 94% | 35 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.012 | 0.019 | 0.019 | +0.0037 | +0.0063 | 74% | 35 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.031 | <0.001 |  | +0.0150 | +0.0179 | 83% | 35 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0018 | 0.840 | -0.0014 | 0.891 | -0.0005 | 0.831 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0171 | 0.078 | -0.0131 | 0.023 | -0.0180 | <0.001 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0046 | 0.565 | +0.0023 | 0.576 | +0.0026 | 0.516 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0214 | 0.023 | -0.0130 | 0.011 | -0.0175 | <0.001 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0103 | 0.039 | +0.0034 | 0.287 | +0.0058 | 0.367 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0149 | 0.219 | -0.0043 | 0.680 | -0.0000 | 0.957 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0106 | 0.007 | +0.0012 | 0.743 | +0.0017 | 0.789 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0188 | 0.158 | -0.0113 | 0.078 | -0.0115 | 0.078 |

Under the gentler schedule (the one for which §5.15 reports the largest gains of the variance-reduction rules) the picture reverses: the graph-regularised design beats Random in all four blocks (log-loss +0.033, +0.039, +0.044, +0.041, Holm p < 0.002, better in 74–97% of seeds; AUC +0.016 to +0.019; accuracy +0.016 to +0.028), and so does `vopt_u` (+0.031 to +0.035); the typed coverage rule is weaker (+0.012 to +0.019). The graph design is not distinguishable from `vopt_u` in split A (+0.002, +0.005) and is above it on log-loss in split B (+0.010 and +0.011, raw p 0.039 and 0.007, uncorrected; AUC +0.003 and +0.001). The shuffled-graph control was not part of this run.

**Part 5: replication of the frozen graph-regularised design under the gentler schedule (40 new seeds, 1700–1739).** Because the frozen `gvopt_lap3` had been fixed before Part 4 and was not tuned on it, a pre-registered replication was run on 40 seeds not used before, with `vopt_u` and the shuffled-graph control `gvopt_lap3_shuffled` (permuted node identities of the graph, same degree sequence). H1: positive log-loss gain over Random in every block (Holm over the four blocks); H2: a positive paired difference to `vopt_u` and to the shuffled control, claimed only if significant after Holm in a split-B block.

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.039 | <0.001 | <0.001 | +0.0159 | +0.0191 | 80% | 40 |
| A | single-shot | (control) same, graph shuffled | +0.038 | <0.001 |  | +0.0169 | +0.0182 | 88% | 40 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.034 | <0.001 |  | +0.0148 | +0.0192 | 88% | 40 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.039 | <0.001 | <0.001 | +0.0174 | +0.0196 | 85% | 40 |
| A | sequential | (control) same, graph shuffled | +0.038 | <0.001 |  | +0.0182 | +0.0175 | 88% | 40 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.036 | <0.001 |  | +0.0179 | +0.0182 | 80% | 40 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.038 | <0.001 | <0.001 | +0.0169 | +0.0175 | 80% | 40 |
| B | single-shot | (control) same, graph shuffled | +0.036 | <0.001 |  | +0.0163 | +0.0201 | 78% | 40 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.038 | <0.001 |  | +0.0184 | +0.0212 | 78% | 40 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.031 | <0.001 | <0.001 | +0.0148 | +0.0184 | 85% | 40 |
| B | sequential | (control) same, graph shuffled | +0.029 | <0.001 |  | +0.0138 | +0.0195 | 78% | 40 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.028 | <0.001 |  | +0.0124 | +0.0184 | 78% | 40 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0053 | 0.221 | +0.0011 | 0.493 | -0.0001 | 0.734 |
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0012 | 0.879 | -0.0011 | 0.485 | +0.0009 | 0.739 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0031 | 0.320 | -0.0005 | 0.765 | +0.0014 | 0.338 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0005 | 0.755 | -0.0008 | 0.444 | +0.0021 | 0.177 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0001 | 0.755 | -0.0015 | 0.675 | -0.0037 | 0.222 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0018 | 0.675 | +0.0006 | 0.858 | -0.0026 | 0.379 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0038 | 0.354 | +0.0024 | 0.301 | +0.0001 | 0.780 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0022 | 0.420 | +0.0010 | 0.538 | -0.0010 | 0.722 |

H1 is met: the graph-regularised design beats Random on log-loss in all four blocks (+0.039, +0.039, +0.038, +0.031; Holm p < 0.001; better in 80–85% of seeds), with AUC gains of +0.015 to +0.017 and accuracy gains of +0.018 to +0.019. H2 is not met: the differences to `vopt_u` (+0.005, +0.003, +0.000, +0.004) and to the shuffled-graph control (+0.001, +0.001, +0.002, +0.002) are within noise (raw p ≥ 0.22), and the shuffled-graph control and `vopt_u` have essentially the same gains over Random as the graph version (+0.036 to +0.038 and +0.028 to +0.038). The split-B advantage over `vopt_u` seen in Part 4 (+0.010, raw p 0.007 and 0.039) did not reproduce. The conclusion is that, under the gentler head schedule, variance-reduction design on the last layer beats Random reproducibly (this section and §5.15), and that adding a graph prior to the design neither helps nor hurts measurably.

**Part 6: the typed coverage rule on 40 further fresh seeds under the re-tuned schedule (seeds 1600–1639), and pooled over all fresh sets.** Because the plain typed coverage rule was significant in Part 1, in two blocks only in Part 3, and its own controls had been run only in Part 1, a pre-registered large-sample run (`OVERNIGHT2_PLAN.md`, P5) repeated it, unchanged, with the type-only and shuffled-posterior controls and `vopt_u` on 40 seeds not used before (primary test: log-loss gain over Random, Holm over the four blocks).

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.023 | 0.008 | 0.008 | +0.0077 | +0.0131 | 68% | 40 |
| A | single-shot | (control) decisive predictor from the type only | -0.007 | 0.368 |  | -0.0051 | -0.0026 | 48% | 40 |
| A | single-shot | (control) type posterior shuffled over images | +0.010 | 0.092 |  | -0.0004 | +0.0050 | 62% | 40 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.014 | 0.206 |  | +0.0076 | +0.0145 | 62% | 40 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.009 | 0.183 | 0.183 | +0.0033 | +0.0085 | 60% | 40 |
| A | sequential | (control) decisive predictor from the type only | -0.015 | 0.048 |  | -0.0073 | -0.0037 | 38% | 40 |
| A | sequential | (control) type posterior shuffled over images | -0.010 | 0.192 |  | -0.0026 | +0.0041 | 40% | 40 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.017 | 0.068 |  | +0.0093 | +0.0156 | 65% | 40 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.029 | <0.001 | <0.001 | +0.0038 | +0.0076 | 70% | 40 |
| B | single-shot | (control) decisive predictor from the type only | +0.019 | 0.019 |  | -0.0040 | +0.0008 | 72% | 40 |
| B | single-shot | (control) type posterior shuffled over images | +0.024 | 0.019 |  | +0.0000 | +0.0044 | 62% | 40 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.001 | 0.452 |  | +0.0018 | +0.0056 | 57% | 40 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.023 | 0.031 | 0.031 | +0.0025 | +0.0072 | 65% | 40 |
| B | sequential | (control) decisive predictor from the type only | +0.018 | 0.013 |  | -0.0023 | +0.0005 | 68% | 40 |
| B | sequential | (control) type posterior shuffled over images | +0.027 | 0.002 |  | +0.0004 | +0.0038 | 72% | 40 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.007 | 0.921 |  | +0.0035 | +0.0173 | 57% | 40 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0300 | 0.001 | +0.0127 | <0.001 | +0.0158 | 0.001 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0133 | 0.183 | +0.0080 | 0.017 | +0.0082 | 0.060 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0088 | 0.320 | +0.0000 | 0.375 | -0.0014 | 0.762 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0241 | 0.004 | +0.0105 | 0.001 | +0.0122 | 0.005 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0183 | 0.032 | +0.0058 | 0.037 | +0.0045 | 0.227 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0086 | 0.745 | -0.0060 | 0.397 | -0.0071 | 0.232 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0107 | 0.237 | +0.0078 | 0.014 | +0.0068 | 0.141 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0052 | 0.618 | +0.0038 | 0.354 | +0.0032 | 0.497 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0281 | 0.037 | +0.0020 | 0.705 | +0.0019 | 0.558 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0049 | 0.675 | +0.0048 | 0.216 | +0.0067 | 0.307 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | -0.0049 | 0.795 | +0.0021 | 0.581 | +0.0034 | 0.503 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0293 | 0.079 | -0.0010 | 0.952 | -0.0100 | 0.077 |

The rule has a positive log-loss gain in all four blocks (+0.023, +0.009, +0.029, +0.023; raw p 0.008, 0.18, 0.001, 0.031; Holm over the four blocks 0.025, 0.18, 0.004, 0.062), so it is significant after correction in two of the four blocks (A single-shot and B single-shot), with positive AUC (+0.002 to +0.008) and accuracy (+0.008 to +0.013) gains throughout. The controls separate the graph from the type: the type-only decisive predictor is at −0.007 and −0.015 in split A and +0.019 and +0.018 in split B, and the rule is above it in AUC in all four blocks (+0.013, +0.011, +0.008, +0.005; raw p < 0.001, 0.001, 0.014, 0.22) and in log-loss in split A (+0.030, +0.024; raw p 0.001, 0.004). Against the shuffled-posterior control the AUC differences are +0.008, +0.006, +0.004 and +0.002 (raw p 0.017, 0.037, 0.35, 0.58) and the log-loss differences +0.013, +0.018, +0.005 and −0.005. In split B the gain over Random is the same for the real and the shuffled posterior (+0.029 against +0.024 single-shot; +0.023 against +0.027 sequential), so in split B the gain is not attributable to the graph; in split A it is, partly. `vopt_u` was at +0.014, +0.017, +0.001 and −0.007 on these seeds, so the typed rule is above it in split B (+0.028 and +0.029, raw p 0.037 and 0.079) and not different in split A, which shows how strongly the gain of the non-graph design varies between seed sets (its gains over the three fresh sets of this section (Parts 1, 3 and 6) range from −0.007 to +0.058).

Pooling all fresh confirmatory seed sets run under the re-tuned schedule (Parts 1, 3 and 6; 105 seeds; a post-hoc summary, not a pre-registered test), the rule beats Random in all four blocks:

```
block             n  log-loss   raw p   Holm4      AUC      acc  better | minus typeonly (n) LL / AUC p | minus shuffled LL / AUC p | minus vopt_u (n) LL p
A-single      105    +0.022  <0.001  <0.001  +0.0073  +0.0088     66% | +0.024/+0.0125 p <0.001/<0.001 (n=70) | +0.010/+0.0077 p 0.143/0.006 | +0.009 (n=105) p 0.240
A-sequential  105    +0.014   0.007   0.007  +0.0044  +0.0090     62% | +0.025/+0.0122 p <0.001/<0.001 (n=70) | +0.016/+0.0068 p 0.013/0.009 | -0.005 (n=105) p 0.568
B-single      105    +0.037  <0.001  <0.001  +0.0088  +0.0092     76% | +0.015/+0.0096 p 0.099/0.001 (n=70) | +0.008/+0.0050 p 0.255/0.048 | +0.012 (n=105) p 0.182
B-sequential  105    +0.032  <0.001  <0.001  +0.0071  +0.0084     68% | +0.015/+0.0090 p 0.084/0.003 (n=70) | +0.007/+0.0061 p 0.288/0.039 | +0.014 (n=105) p 0.275
```

(Columns: pooled log-loss gain over Random, raw p and Holm over four blocks, AUC and accuracy gain, share of seeds better; then the paired difference to the type-only control and to the shuffled-posterior control, both from the 70 seeds of Parts 1 and 6, as log-loss and AUC differences with raw p-values; then the difference to `vopt_u` over all 105 seeds.) The pooled gains are +0.014 to +0.037 in log-loss, with Holm p < 0.01 in each block. The type posterior adds to the type-only predictor in AUC in every block (+0.009 to +0.013, p ≤ 0.003) and in log-loss in split A (+0.024, +0.025, p < 0.001), and adds to the shuffled posterior in AUC (+0.005 to +0.008, raw p 0.006 to 0.048); the log-loss differences to the shuffled posterior are not significant in three of four blocks. These are the strongest signs in this report that the graph-derived type information carries something beyond the type itself, and they are small (AUC +0.005 to +0.013) and uncorrected for multiplicity.


**Summary of the graph work (§5.16–5.20).** (i) The comparison graph is nearly a matching, so a ranking network on it has nothing to recover. (ii) With the old head schedule no graph rule beat Random reproducibly (§5.16–5.18). (iii) With the re-tuned schedule the frozen type-aware graph rules beat Random in 4 of 4 blocks on 30 fresh seeds and in the two split-B blocks on 35 further seeds; the type posterior adds AUC over a type-only decisive predictor in Part 1, but not distinguishably from the non-graph variance-reduction rule `vopt_u`. (iv) With the gentler schedule the graph-regularised variance-reduction design beats Random in 4 of 4 blocks on two independent sets of 35 and 40 seeds, but its shuffled-graph control and `vopt_u` do the same, so there is no evidence that the graph is responsible for the gain. (v) Selection of a method on 25 development seeds repeatedly overstated its effect (the Laplacian prior in split A, Part 3), so only the pre-registered confirmations should be read as evidence. **Limitations:** one encoder (frozen SimCLR), the same 168 pair groups in every seed (so each run tests sensitivity to the split draw, not new data, and Wilcoxon p-values are optimistic), three head schedules (old, re-tuned, gentle) with conclusions that depend on the schedule, modest effects (log-loss +0.02 to +0.05, AUC +0.005 to +0.019), paired differences to controls that are not corrected for multiplicity, and a graph built from the same frozen features as the head, which limits how much new information it can add.

## 6. Conclusion

The implemented protocol separates pair-disjoint acquisition groups, SHA-256 content-identity exclusion of outer-test images, validation-only training decisions, and artifact-level auditing, and its results change what can be claimed about acquisition strategies.

*Fine-tuned reward model (§5.8–5.10).* Task 3b (validation-selected schedules) and Task 3c (eight strategies, five budgets, five seeds) are complete. Leaders differ by budget, but the gaps are small relative to seed variance and to training noise that depends on the order of the selected pairs; at budget 100 all strategies train on identical data. The pre-registered fixed-epoch comparison (§5.9) shows 30 epochs exceeding 3 epochs at budgets of 25 and above, except for budget-75 Uncertainty (0.357 for both). No acquisition strategy can be recommended from these runs.

*Frozen encoder (§5.11).* Caching the SimCLR features and training only the reward head, with an order-invariant loss, removes most training noise (residual head-initialisation sd about 0.04) and lifts type accuracy to 0.73–0.86, in line with a frozen-encoder nearest-neighbour baseline (0.871 ± 0.041). But the reference-anchor term alone reaches 0.848 with zero pair groups, pair labels alone reach 0.37–0.55, and the 10 initial groups already reach the plateau: type accuracy on ideal images is not an endpoint on which acquisition rules can be compared.

*Held-out preference endpoint (§5.12).* Across 33 strategy variants and 35 seeds, the evidence for any acquisition rule beating random selection is limited. With one batch chosen from the 10-group model, Laplace BALD and BALD × P(decisive) are significantly better than Random on log-loss and AUC after correction (gains of about 0.04 and 0.01), and the original Cluster-quota uncertainty rule and a DPP selector on log-loss only (the Cluster-quota advantage does not survive scoring all heads instead of the first row's type); with sequential retraining none is significant, and a second training schedule changes which strategies are significant on log-loss and leaves only Laplace BALD's AUC gain. Uncertainty-driven rules (Uncertainty, MC-dropout, BADGE-style, DropQuery) are no better than Random in either condition, graph-cut selection is worse in both, and the choice of initial set does not matter detectably. With the 20% pair-level hold-out of the classifier2 work (§5.13), the significant rule is instead plain core-set (+0.052 log-loss single-shot and +0.053 sequential, log-loss only) and neither Laplace BALD nor BALD × P(decisive) is significant, so no rule is significant under both splits; coverage-oriented rules lead at the smallest budgets in both. A ranking that looked decisive on five seeds (Cluster-Margin, core-set) shrank to nothing over 35, a caution for any comparison that rests on a few seeds and a few dozen test images.

*Judgment as the query unit (§5.14).* The comparisons above counted the budget in pair groups and revealed every judgment of a selected group, which does not match the labelling software (one pair and one type per query). Repeating them with one (pair, type) judgment as the query unit and the budget in judgments (10–60) leaves the main conclusion unchanged: on log-loss, calibrated log-loss and AUC no variant is significantly better than Random after Holm correction in three of the four split-by-condition tables; in Split B sequential an all-head Cluster-quota rule (+0.074 log-loss) and deep-ensemble BALD × P(decisive) (+0.014 AUC) are, and DPP and all-head Uncertainty are significantly worse in Split B single-shot. The significant rules of §5.12–5.13 (Laplace BALD, BALD × P(decisive), core-set) are not significant here, so which rule helps depends on the query unit as well as on the split. With a head schedule recalibrated for the judgment unit (gentler: lr 0.001–0.003), Random's log-loss improves by 0.03–0.14 and the two positive primary-metric results above vanish, no variant beats Random on log-loss, and uncertainty-driven rules become significantly worse than Random in Split B single-shot. Decisive-judgment accuracy is 1–3 points higher than Random for deep-ensemble BALD × P(decisive) and Fisher D-optimal design in all four cells, and this replicated in a pre-registered run on 35 new seeds (same 168 groups; deep-ensemble BALD alone in 3 of 4 cells), whereas the log-loss effects of individual rules shrank or vanished (all-head Cluster-quota, core-set, DPP) and the AUC gain of deep-ensemble BALD × P(decisive) in Split B sequential replicated (+0.014). The gain is modest, appears for rules that choose the type with the model, and has not been tested on new data or with a random initial set at 35 seeds. The result does not show that counting groups had hidden a strategy effect, and it is limited by small test sets, one encoder and one head schedule.

*Variance-reduction design under a suitable head schedule (§5.15).* The conclusion that no rule beats Random holds for the old head schedule and for the log-loss of the re-tuned schedule on the 35 main seeds; for ranking metrics it does not hold once the head is trained with a schedule suited to small label sets. In five pre-registered confirmatory runs on fresh splits, greedy I-optimal design of the last layer (and the earlier Fisher D-optimal and BALD × P(decisive) rules) improved decisive accuracy by 0.010–0.017 and AUC by 0.006–0.013 in the main protocol (the first run reached significance on accuracy only, the replication on all eight tests), by 0.019–0.024 accuracy and 0.017–0.020 AUC from a cold start (all 12 tests), and a soft HTR priority weight raised HTR accuracy by 0.067 and HTR AUC by 0.058 without a price on the other types. Plain uncertainty sampling and relation-aware core-set do not share the effect, the new rule is only modestly and not uniformly ahead of Fisher and BALD × P(decisive), the effect holds in a cross-world replication on image-disjoint halves (pre-registration 6) but depends on the half, and it decays with the number of labels already in hand (accuracy gain +0.019 from 10 random judgments, +0.017 with about 30, +0.007 with 20 groups, none from 40–60 groups); all of this is conditional on the 168 pair groups, so a prospective test on the lab's new growth data is the decisive next experiment.

*Controls on the ideal-image endpoint (§5.19).* Simple controls (raw pixels, random-weight ResNet-18, hand-made peak features, frozen SimCLR with 1-NN or logistic regression, fine-tuned ResNet-18) reach 0.81-0.91 accuracy on the 28-image outer test with the same references, indistinguishable from one another and far above the fine-tuned reward model of §5.10; this endpoint cannot rank encoders or acquisition rules.

*Graph-aware selection (§5.16–5.20).* The graph of the comparisons is nearly a matching (168 edges over 284 images, 117 components; per type at most 4–5 connected images and 0–6 images that both won and lost), so a model that recovers a ranking from it has no structure to use. On an image-similarity graph, with the old head schedule, first-generation rules (PageRank- or boundary-weighted uncertainty, a graph-propagated core-set) showed nothing significant on 35 seeds, and type-aware rules (label propagation of the typed reference images over the graph, a predicted probability that the judgment is decisive, farthest-first coverage in graph-propagated pair space) that looked strong on 10 development seeds failed the pre-registered criterion on 25 unseen seeds and replicated only weakly. With the re-tuned head schedule the same frozen type-aware coverage rule beats Random reproducibly (all four blocks on 30 fresh seeds; pooled over 105 fresh seeds, log-loss +0.014 to +0.037, Holm p < 0.01 in each block, AUC +0.004 to +0.009), and the graph-derived type posterior adds a small amount over a type-only decisive predictor (AUC +0.009 to +0.013, log-loss +0.024 in split A) and over a shuffled posterior (AUC +0.005 to +0.008), but it is not distinguishable from the non-graph variance-reduction design of §5.15. A graph-regularised variance-reduction design beats Random under the gentler schedule on two independent seed sets, but its shuffled-graph control and the non-graph design do the same. The supported statement is therefore: graph-based type information gives a small, reproducible gain over random selection under a suitable head schedule, comparable to the best non-graph rule, and graph regularisation of the design has no measurable effect beyond it; method selection on 25 development seeds overstated effects twice, so only the pre-registered confirmations count.

*Next steps.* The held-out preference endpoint, not ideal-image type accuracy, should be the primary measure for deciding which pairs to label; coverage rules (core-set) and the Fisher/Laplace-posterior rules are the natural candidates to confirm with a larger candidate pool or a second encoder (ImageNet, or a larger self-supervised model), and the choice of query unit now follows the labelling protocol (§5.14). Data freeze manifests are generated by `generate_data_freeze_manifest.py` and fail loudly on any overlap violation or unexpected partition size. Metadata fusion (§5.6) and trajectory integration (§5.7) remain deferred pending structured process-variable data and a validated five-state classifier, respectively.

## References

1. R. A. Bradley and M. E. Terry. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*. Biometrika, 1952.
2. B. Settles. *Active Learning Literature Survey*. University of Wisconsin--Madison, 2009.
3. K. He, X. Zhang, S. Ren, and J. Sun. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
4. T. Chen et al. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML, 2020.
5. G. Citovsky, G. DeSalvo, C. Gentile, L. Karydas, A. Rajagopalan, A. Rostamizadeh, and S. Kumar. *Batch Active Learning at Scale*. NeurIPS, 2021.
6. Y. Gal and Z. Ghahramani. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML, 2016.
7. D. Lischuk and S. Dasgupta. *Adaptive Sampling for Estimation of a Probability Distribution*. ICML, 2017. *(See also Yang and Loog, 2016, for uncertainty-diversity acquisition.)*

## Appendix A. Figure Provenance

- Figure 1: generated by `active_learning_studies/pair_disjoint_not_image_disjoint/generate_academic_report_assets.py`.
- Figure 2: generated by the same script from the implemented split contract.
- Figure 3: generated by the script from the cited historical CSV; it introduces no new numerical results.
- Figure 4: generated by the script from the Stage 2 symmetry-factorial aggregate CSV.
- Figure 5: generated by the script from the encoder-screen validation CSV.
- Figures 6--9: existing PCA/t-SNE artifacts from `active_learning_studies/image_representation_analysis/results/representation_exploration/`; they use the exploratory manifest and are not active-learning outcome figures.
- Figures 10--12: generated by `generate_section5_representation_diagnostics.py` from saved exploratory coordinate CSVs.
- Figure 13: generated from `strategy_followup_analysis/fifteen_seed_extension/fifteen_seed_summary.csv`.
- Figure 14: generated from `strategy_followup_analysis/redundancy_analysis/redundancy_utility_accuracy.csv`.
- Figure 15: generated from stored selected pair embeddings in the historical follow-up analysis.
- Figure 19: generated by `active_learning_studies/pair_disjoint_not_image_disjoint/plot_endpoint_forest.py` from `results/pair_endpoint_study/{single,sequential}_all_*_vs_random.csv` and the per-seed cells in `results/pair_endpoint_study/{cells,sequential_cells}/`.
- Tables of §5.15 (pre-registrations 1–6): generated by `new_methods_final_tables.py` and `new_methods_confirm6.py` (`results/new_methods/CONFIRMATORY_TABLES.md`) from `results/new_methods/{confirm,confirm2,confirm3_coldstart,confirm4_htr,confirm5_priority}/`; the strategy code is `new_methods_strategies.py`, the runner `new_methods_run.sh`; pre-registrations `NEW_METHODS_PREREGISTRATION*.md`.
- Table 1: generated by `generate_ideal_split_capacity_summary.py` from the default identity-safe split contract and current ideal-image files.
- Table of §5.19: parsed from `classification_baselines/results/table_paper_seeds_with_finetune.md` (generated by `run_baselines.py`, `finetune_baseline.py`, `aggregate_baselines.py`); the full tables and checks are in `classification_baselines/README.md`.
- Tables of §5.20 (graph rules under the re-tuned and gentler schedules): generated by `make_overnight2_tables.py`, `make_overnight2_section.py` and `pool_typed.py` from `results/new_methods/graph_p{1,2,3,4,5,6}/{A,B}/{single,sequential}/seed*_*.json` (aggregators `aggregate_overnight2.py`, `choose_p2.py`); plan, pre-registrations and log: `graph_exploration/OVERNIGHT2_PLAN.md`, `AUTONOMOUS_RUN_LOG.md`.
- Tables of §5.16–5.18 (graph-aware selection): the first-generation tables by `aggregate_graph_vs_random.py` and `make_graph_report_section.py`, the confirmation by `aggregate_confirm.py` (bootstrap intervals, Holm over 12 tests) and `aggregate_confirm_perblock.py` (Holm per block), the 20-seed replication by `aggregate_phase2.py`, the 30-seed split-B replication by `aggregate_replication.py`, all from `results/judgment_unit_study/{A_groups,B_groups}/{single,sequential}/seed*_*.json`; the run logs, pre-registrations and design/literature notes are in `graph_exploration/` (`AUTONOMOUS_RUN_LOG.md`, `DESIGN_AND_LITERATURE.md`).
- Tables of §5.14: generated by `judgment_unit_aggregate.py` and `judgment_unit_tables.py` from `results/judgment_unit_study/{A_groups,B_groups,A_random,B_random}/`; the full tables, adaptation list and checks are in `JUDGMENT_UNIT_RESULTS.md`.

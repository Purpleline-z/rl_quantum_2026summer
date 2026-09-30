# 3. Method

## 3.1 Problem Formulation

Let $\mathcal{X}$ denote the set of $N$ RHEED images collected during molecular beam epitaxy growth experiments. Each image can be associated with one of $K = 5$ surface reconstruction types, drawn from the index set $\mathcal{K} = \{1, \ldots, K\}$. The five reconstruction types correspond to distinct surface periodicities observable in RHEED patterns: $(1\times1)$, Twinned$(2\times1)$, $c(6\times2)$, $(\sqrt{13}\times\sqrt{13})$, and HTR.

We adopt a pairwise preference learning framework. The unlabeled candidate pool is a collection of triples

$$\mathcal{U} \subset \mathcal{X} \times \mathcal{X} \times \mathcal{K},$$

where each element $(i, j, k)$ represents an image pair with known reconstruction type $k$. A human annotator observes the pair and indicates which image exhibits higher reconstruction quality, yielding a binary label $y \in \{0, 1\}$ (where $y = 1$ denotes that image $i$ is preferred over image $j$). The labeled set $\mathcal{L}$ is initialized to the empty set; at each active learning round $t$, a batch of $B$ triples is selected from $\mathcal{U}$, annotated, and moved to $\mathcal{L}$.

**Model.** We parameterize a multi-head reward model $r: \mathcal{X} \to \mathbb{R}^K$ that maps each image to $K$ scalar reward scores, one per reconstruction type. The architecture is a ResNet-18 encoder (pretrained with SimCLR contrastive learning) followed by a shared reward head: $\text{Linear}(512 \to 256) \to \text{ReLU} \to \text{Dropout}(p=0.1) \to \text{Linear}(256 \to 5)$. The $k$-th output $r_k(i;\theta)$ is the reward assigned to image $i$ under reconstruction type $k$.

**Bradley-Terry likelihood.** For a pair $(i, j)$ of type $k$, we model the probability that $i$ is preferred over $j$ via the Bradley-Terry model:

$$P(\text{$i$ beats $j$} \mid k;\theta) = \sigma\!\left(r_k(i;\theta) - r_k(j;\theta)\right),$$

where $\sigma(z) = (1 + e^{-z})^{-1}$ is the sigmoid function. Let $\Delta r_k(i,j;\theta) = r_k(i;\theta) - r_k(j;\theta)$ denote the reward difference for type $k$.

**Training objective.** Given the labeled set $\mathcal{L}$, model parameters $\theta$ are learned by minimizing the negative log-likelihood under the Bradley-Terry model:

$$\ell(\theta) = -\sum_{(i,j,k,y)\in\mathcal{L}} \left[ y \log \sigma(\Delta r_k) + (1 - y)\log\!\left(1 - \sigma(\Delta r_k)\right) \right],$$

where $\Delta r_k = \Delta r_k(i,j;\theta)$ is abbreviated for clarity. This loss decomposes cleanly across reconstruction types, providing an independent training signal for each of the $K$ reward heads.

---

## 3.2 Acquisition Functions

At each active learning round, we score every unlabeled candidate $(i,j,k) \in \mathcal{U}$ with an acquisition function $a(i,j,k)$ and select the top-$B$ triples (or a batch derived therefrom) for annotation. We describe six acquisition strategies, distinguishing our type-conditioned variants from type-agnostic baselines.

**Random (baseline).** As a lower bound on performance, pairs are selected uniformly at random from $\mathcal{U}$ with no reference to model predictions:

$$a_\text{rand}(i,j,k) = \text{Uniform}(\mathcal{U}).$$

**Type-Agnostic Uncertainty (baseline).** A natural uncertainty-sampling baseline computes predictive entropy for each head and averages over all $K$ heads, ignoring the known reconstruction type $k$:

$$a_\text{avg}(i,j,k) = -\frac{1}{K}\sum_{k'=1}^{K} H\!\left(\sigma(\Delta r_{k'})\right),$$

where $H(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy function. We negate entropy so that higher scores indicate less uncertainty, and acquisition selects the most negative values (i.e., the highest-entropy pairs). This baseline treats all heads symmetrically and discards the type label $k$.

**Type-Conditioned Uncertainty (ours).** When the reconstruction type $k$ of a pair is known — as it always is in our setting — it is wasteful to average over all $K$ heads. We instead query only the head responsible for type $k$:

$$a_\text{tc}(i,j,k) = -H\!\left(\sigma(\Delta r_k)\right).$$

This acquires pairs for which the model's prediction under the relevant head is maximally uncertain, i.e., $\sigma(\Delta r_k) \approx 0.5$. Section 3.3 formalizes why this yields a tighter proxy for epistemic uncertainty.

**MC-BALD, Type-Conditioned (ours).** To disentangle epistemic from aleatoric uncertainty, we apply Bayesian Active Learning by Disagreement (BALD; Houlsby et al., 2011) using Monte Carlo dropout. Let $\omega \sim q(\omega)$ denote a stochastic dropout mask and $\Delta r_k^\omega$ the reward difference under mask $\omega$. We draw $S = 20$ forward passes and compute:

$$a_\text{bald}(i,j,k) = H\!\left(\mathbb{E}_\omega\!\left[\sigma(\Delta r_k^\omega)\right]\right) - \mathbb{E}_\omega\!\left[H\!\left(\sigma(\Delta r_k^\omega)\right)\right].$$

The first term is the entropy of the expected prediction (total uncertainty), and the second is the expected entropy of individual predictions (expected aleatoric uncertainty). Their difference measures mutual information between the model parameters and the label, restricted to head $k$. Applying type-conditioning here prevents high-aleatoric heads from inflating the epistemic signal.

**Cluster-Margin (Citovsky et al., 2021).** To encourage diversity alongside uncertainty, we adapt the Cluster-Margin strategy of Citovsky et al. (2021). We first prefilter $\mathcal{U}$ to the $2B$ candidates with the smallest margin under the relevant head:

$$m(i,j,k) = \left|\sigma(\Delta r_k) - 0.5\right|.$$

The retained candidates are then assigned to clusters (obtained by $k$-means on the cached image embeddings), and $B$ pairs are selected by round-robin over clusters. This balances informativeness (low margin) with representativeness (cluster diversity). Our implementation applies type-conditioning to the margin computation, using head $k$ rather than a head-averaged margin.

**Fisher Information, Type-Conditioned (ours).** To account for the geometry of the parameter space, we define an acquisition function based on the expected information gain in the Fisher sense. Let $\mathbf{F}_k \in \mathbb{R}^{d \times d}$ be the accumulated Fisher information matrix for the parameters of head $k$, maintained incrementally across rounds. For a candidate pair $(i, j, k)$, let

$$\mathbf{g}_k = \nabla_\theta r_k(i;\theta) - \nabla_\theta r_k(j;\theta)$$

be the gradient of the reward difference with respect to the model parameters. The expected information gain from annotating this pair is:

$$a_\text{fisher}(i,j,k) = \log\det\!\left(\mathbf{F}_k + \sigma(\Delta r_k)\bigl(1 - \sigma(\Delta r_k)\bigr)\,\mathbf{g}_k\mathbf{g}_k^\top\right) - \log\det(\mathbf{F}_k).$$

The scalar weight $\sigma(\Delta r_k)(1 - \sigma(\Delta r_k))$ is the Fisher information of a Bernoulli observation at the current predictive probability, and $\mathbf{g}_k\mathbf{g}_k^\top$ is the rank-one outer product capturing the direction in parameter space probed by the pair. By the matrix determinant lemma, this reduces to $\log(1 + \sigma(\Delta r_k)(1-\sigma(\Delta r_k))\,\mathbf{g}_k^\top \mathbf{F}_k^{-1}\mathbf{g}_k)$, enabling efficient computation without a full determinant solve. Type-conditioning isolates the gradient to head $k$, ensuring that the Fisher update for one reconstruction type does not confound the acquisition signal for another.

---

## 3.3 Proposition 1 — Type-Conditioning Reduces Acquisition Noise

We now provide a formal justification for why type-conditioned acquisition is preferable to type-agnostic averaging.

**Proposition 1.** *Let $p_k = \sigma(\Delta r_k(i,j;\theta))$ be the win probability under head $k$ for a fixed pair $(i,j)$, and let $\bar{p} = \frac{1}{K}\sum_{k=1}^K p_k$ be the mean win probability averaged over all heads. Then:*

$$H\!\left(\frac{1}{K}\sum_{k=1}^{K} p_k\right) \;\geq\; \frac{1}{K}\sum_{k=1}^{K} H(p_k).$$

*Consequently, the type-agnostic acquisition score $-\frac{1}{K}\sum_{k'} H(\sigma(\Delta r_{k'}))$ systematically underestimates the magnitude of the type-agnostic entropy $H(\bar{p})$, and the type-agnostic entropy overestimates the uncertainty of the pair relative to the type-conditioned head by at most $\log 2$ nats (equivalently, $1$ bit). The type-conditioned acquisition $a_\text{tc}$ removes this overestimation and provides a tighter proxy for epistemic uncertainty on the relevant head $k$.*

**Proof sketch.** The inequality follows directly from Jensen's inequality applied to the binary entropy function $H: [0,1] \to [0, \log 2]$. Since $H$ is concave on $[0,1]$ — its second derivative is $-1/(p(1-p)) < 0$ everywhere on the interior — we have for any distribution over $\{p_k\}_{k=1}^K$:

$$H\!\left(\frac{1}{K}\sum_{k=1}^K p_k\right) \;\geq\; \frac{1}{K}\sum_{k=1}^K H(p_k).$$

Now consider the type-agnostic acquisition score. It computes the average head entropy $\frac{1}{K}\sum_{k'} H(\sigma(\Delta r_{k'}))$, which by Jensen's inequality is *at most* $H(\bar{p})$. This average can be strictly less than $H(p_k)$ when the non-type-$k$ heads $\{p_{k'}: k' \neq k\}$ have low entropy (i.e., those heads are already confident), pulling the average down and causing a genuinely uncertain pair under head $k$ to receive a low acquisition score. In the extreme case, if all $K - 1$ irrelevant heads have $p_{k'} \in \{0,1\}$ (maximal confidence), they contribute $H(p_{k'}) = 0$ and the average is $\frac{1}{K}H(p_k)$, suppressing the true uncertainty by a factor of $K$.

More formally, the difference $H(p_k) - \frac{1}{K}\sum_{k'=1}^K H(p_{k'})$ can be as large as $\frac{K-1}{K}\log 2$ nats per pair. Over a budget of $B$ pairs, the type-agnostic method may therefore rank genuinely uncertain pairs lower than confident pairs whose irrelevant heads happen to disagree, introducing up to $\log K$ nats of noise into the acquisition ranking. The type-conditioned acquisition $a_\text{tc}(i,j,k) = -H(p_k)$ eliminates this confound by construction, using only the head indexed by the known reconstruction type $k$. $\square$

---

## 3.4 Embedding Cache and Computational Efficiency

A practical concern in active learning is acquisition overhead: re-evaluating all candidates after each round should not dominate total training time. We address this with a frozen-encoder embedding cache.

Because the ResNet-18 encoder is pretrained with SimCLR and held frozen during reward learning (only the reward head is fine-tuned), image embeddings $\phi(x) \in \mathbb{R}^{512}$ are computed exactly once over the full image corpus $\mathcal{X}$ and stored on CPU memory. Each embedding requires approximately $512 \times 4$ bytes $\approx 2$ KiB, so the full cache for $N$ images occupies $\sim 2N$ KiB and easily fits in RAM for datasets of the scale considered here ($N \leq 10^4$).

All acquisition scores are then computed from cached embeddings passed through the reward head only, bypassing the $\sim\!11$M parameters of the encoder. For the uncertainty-based methods, the per-round complexity is $O(|\mathcal{U}| \cdot K \cdot T)$ where $T$ is the number of reward head forward passes required (one for $a_\text{tc}$, $K$ for $a_\text{avg}$). For MC-BALD with $S = 20$ dropout samples, the complexity is $O(|\mathcal{U}| \cdot K \cdot S)$. The Fisher information method requires gradient computation through the reward head, giving $O(d^2 \cdot |\mathcal{U}|)$ per round where $d$ is the number of reward head parameters ($d \approx 132$K); the accumulated Fisher $\mathbf{F}_k$ is maintained incrementally and updated with a rank-one correction at each round, avoiding repeated recomputation from scratch.

Cluster-Margin additionally requires running $k$-means over the candidate embeddings, which we perform once per round on the cached $\phi(\cdot)$ values in $O(|\mathcal{U}| \cdot B \cdot C)$ iterations, where $C$ is the number of clusters.

---

## 3.5 Controlled Synthetic Benchmark

Before evaluating acquisition strategies on real RHEED images — where ground-truth reward orderings are unavailable — we validate all methods on a controlled synthetic benchmark that affords exact evaluation.

**Design.** We sample $N$ images from a standard Gaussian image space and assign each image a ground-truth reward vector $r^*_k(i) \sim \mathcal{N}(0, \sigma_r^2)$ for each type $k$, independently across types and images. Given a pair $(i,j)$ of type $k$, the ground-truth win probability is $p^*_{ij,k} = \sigma(r^*_k(i) - r^*_k(j))$, and labels are sampled accordingly. This construction allows us to directly evaluate the quality of learned rewards via Kendall's $\tau$ correlation between predicted and ground-truth reward rankings:

$$\tau_k = \frac{|\{(i,j): \text{sgn}(r_k(i;\hat\theta) - r_k(j;\hat\theta)) = \text{sgn}(r^*_k(i) - r^*_k(j))\}| - |\{\ldots \neq \ldots\}|}{\binom{N}{2}},$$

reported per-type and averaged over types.

**Factors varied.** We systematically vary three experimental factors: (i) dataset size $N \in \{50, 100, 200, 500\}$, controlling the number of possible candidate pairs $\binom{N}{2}$; (ii) the number of reconstruction types $K \in \{2, 3, 5\}$, to isolate the effect of multi-head interference in type-agnostic acquisition; and (iii) reward separability, parameterized by $\sigma_r \in \{0.5, 1.0, 2.0\}$, which controls the signal-to-noise ratio of pairwise preferences (low $\sigma_r$ produces near-uniform win probabilities, making all pairs similarly uninformative). Each condition is replicated five times with different random seeds.

**Purpose.** The synthetic benchmark isolates the statistical properties of acquisition functions from confounders present in real data (annotation inconsistency, image distribution shift, encoder quality). Strategies that outperform baselines in this controlled setting are considered validated before proceeding to RHEED experiments, where we evaluate using held-out human preference accuracy as a proxy for reward quality.

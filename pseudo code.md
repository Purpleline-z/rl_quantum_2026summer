# Bradley–Terry Reward Model for RHEED Reconstruction Classification

## Algorithm 1: Training Procedure

### Input
- Pairwise comparison dataset $\mathcal{D}_{pair}$
- Ideal reference image dataset $\mathcal{D}_{ideal}$
- Bad-image dataset $\mathcal{D}_{bad}$
- Pretrained SimCLR encoder
- Number of epochs $T$

### Output
- Trained reward model $\phi_\theta$

### Procedure

```text
Initialize reward model φθ:
    ResNet18 encoder + reward head

for epoch = 1 to T do

    for each minibatch do

        Sample training examples from:
            • Pairwise comparisons (60%)
            • Ideal-reference anchors (25%)
            • Bad-image pairs (15%)

        Compute reward scores:
            r1 ← φθ(image1)
            r2 ← φθ(image2)

        Select rewards corresponding to the target
        reconstruction type:
            r1,t , r2,t

        Compute loss:

        if image1 preferred:
            L = −log σ(r1,t − r2,t)

        else if image2 preferred:
            L = −log σ(r2,t − r1,t)

        else if tie:
            L = |r1,t − r2,t|

        else if not applicable:
            L = ReLU(r1,t) + ReLU(r2,t)

        Compute confidence-weighted loss L

        Update parameters using AdamW
        Clip gradients

    end for

    Update learning rate scheduler

end for

Return trained model φθ
```

---

## Bradley–Terry Preference Model

For a pair of images $I_a$ and $I_b$, the probability that image $I_a$ is preferred over image $I_b$ is

$$ P(I_a \succ I_b) = \sigma(r(I_a) - r(I_b)) $$

where

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

is the sigmoid function and $r(\cdot)$ denotes the reward score predicted by the model.

The Bradley–Terry loss is

$$ \mathcal{L}_{BT} = -\log \sigma(r\_{\text{chosen}} - r\_{\text{rejected}}) $$

---

## Model Architecture

```text
Input RHEED Image
        │
        ▼
┌─────────────────────┐
│  ResNet-18 Encoder  │
│ (SimCLR pretrained) │
└─────────────────────┘
        │
        ▼
    512-d Feature
        │
        ▼
┌─────────────────────┐
│     Reward Head     │
│ Linear(512 → 256)   │
│ ReLU                │
│ Dropout(0.1)        │
│ Linear(256 → 5)     │
└─────────────────────┘
        │
        ▼
Five Reconstruction
Reward Scores
```

---

## Additional Explanation

Rather than directly predicting reconstruction labels, the proposed method learns a reward function for each reconstruction type using the Bradley–Terry preference-learning framework. A shared ResNet-18 encoder, initialized with SimCLR self-supervised pretraining on RHEED images, extracts image representations that are mapped to five reconstruction-specific reward scores. Human pairwise comparisons provide ranking supervision, encouraging preferred images to receive higher rewards than non-preferred images. To improve robustness under limited labeled data, ideal reference images are incorporated as anchor examples while low-quality images are used as negative supervision. This formulation allows the model to capture nuanced expert preferences and reconstruction quality differences that are difficult to represent with conventional multiclass classification.
```

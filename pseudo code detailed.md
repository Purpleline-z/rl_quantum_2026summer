
# Algorithm 1: Bradley-Terry Reward Model Training

---

## Input

| Symbol | Description |
|--------|-------------|
| $\mathcal{D}_{\text{pair}}$ | Set of pairwise comparisons $\{(x_i, y_i, y_i', z_i)\}$ where $z_i$ indicates preferred response |
| $\mathcal{D}_{\text{ideal}}$ | Set of ideal reference images with known reconstruction types |
| $\mathcal{D}_{\text{bad}}$ | Set of low-quality images |
| $\phi_\theta$ | Reward model with parameters $\theta$ (ResNet-18 encoder + reward head) |
| $T$ | Number of training epochs |
| $B$ | Batch size |
| $\eta$ | Learning rate |

---

## Output

| Symbol | Description |
|--------|-------------|
| $\theta^*$ | Optimized model parameters |

---

## Procedure

**Step 1: Initialization**
```
1.1  Load pretrained SimCLR encoder weights → ϕ_encoder
1.2  Initialize reward head: Linear(512 → 256) → ReLU → Dropout(0.1) → Linear(256 → 5)
1.3  θ ← (ϕ_encoder, ϕ_head)
1.4  Set optimization hyperparameters: optimizer = AdamW(θ, lr=η, weight_decay=1e-4)
```

**Step 2: Main training loop**
```
For epoch = 1 to T:
    
    For step = 1 to samples_per_epoch:
        
        // 2.1 Sample data source
        r ∼ Uniform(0, 1)
        
        If r < 0.60:
            batch ← SamplePairwise( D_pair )
        Else If r < 0.85:
            batch ← SampleIdealAnchoring( D_ideal, D_traj )
        Else:
            batch ← SampleBadImage( D_bad, D_traj )
        
        // 2.2 Forward pass
        r₁ ← ϕ_θ( batch.img₁ )      // reward scores for image 1
        r₂ ← ϕ_θ( batch.img₂ )      // reward scores for image 2
        
        // 2.3 Extract scores for specific reconstruction type
        r₁* ← r₁[ : , batch.type_idx ]
        r₂* ← r₂[ : , batch.type_idx ]
        
        // 2.4 Compute loss
        L ← ComputeBradleyTerryLoss( r₁*, r₂*, batch.winner, batch.weight )
        
        // 2.5 Backward pass and parameter update
        θ ← θ - η ⋅ ∇_θ L
```

**Step 3: Return**
```
Return θ*
```

---

## Algorithm 2: Bradley-Terry Loss Computation

**Input:** 
- $r_1, r_2 \in \mathbb{R}^B$: reward scores for two images
- $w \in \{0,1,2,3\}^B$: winner labels (0=img1 wins, 1=img2 wins, 2=tie, 3=NA)
- $\rho \in \mathbb{R}^B$: confidence weights

**Output:** $\mathcal{L} \in \mathbb{R}$

```
Procedure ComputeBradleyTerryLoss(r₁, r₂, w, ρ):
    
    L ← 0
    
    // Case 1: Image 1 wins (w = 0)
    mask₁ ← (w = 0)
    If any(mask₁):
        L ← L + E[ -log σ(r₁[mask₁] - r₂[mask₁]) ⋅ ρ[mask₁] ] ⋅ |mask₁| / B
    
    // Case 2: Image 2 wins (w = 1)
    mask₂ ← (w = 1)
    If any(mask₂):
        L ← L + E[ -log σ(r₂[mask₂] - r₁[mask₂]) ⋅ ρ[mask₂] ] ⋅ |mask₂| / B
    
    // Case 3: Tie (w = 2)
    mask_tie ← (w = 2)
    If any(mask_tie):
        L ← L + E[ |r₁ - r₂| ⋅ ρ ] ⋅ |mask_tie| / B
    
    // Case 4: Not applicable (w = 3)
    mask_na ← (w = 3)
    If any(mask_na):
        L ← L + E[ max(0, r₁) + max(0, r₂) ⋅ ρ ] ⋅ |mask_na| / B
    
    Return L
```

where $\sigma(u) = \frac{1}{1 + e^{-u}}$ is the sigmoid function.

---

## Algorithm 3: Pairwise Data Sampling

**Input:** $\mathcal{D}_{\text{pair}}$ (dataframe of pairwise comparisons)

**Output:** Batch of training examples

```
Procedure SamplePairwise( D_pair ):
    
    // 3.1 Randomly select a comparison
    row ← UniformRandom( D_pair )
    
    (path₁, path₂, rt, winner, confidence) ← ExtractFields(row)
    
    // 3.2 Confidence weighting
    If confidence = "Confident":
        weight ← 1.0
    Else If confidence = "Somewhat sure":
        weight ← 0.7
    Else:
        weight ← 1.0
    
    // 3.3 Data augmentation: random swap with 50% probability
    If UniformRandom(0,1) < 0.5:
        Swap(path₁, path₂)
        If winner = "1":
            winner ← "2"
        Else If winner = "2":
            winner ← "1"
    
    // 3.4 Load and transform images
    img₁ ← LoadAndTransform(path₁)  // returns [3, 224, 224] tensor
    img₂ ← LoadAndTransform(path₂)
    
    // 3.5 Convert to indices
    type_idx ← TYPE_TO_IDX[rt]
    winner_idx ← WINNER_MAP[winner]   // 0,1,2,3
    
    Return (img₁, img₂, type_idx, winner_idx, weight)
```

---

## Algorithm 4: Ideal Anchoring Sampling

**Input:** 
- $\mathcal{D}_{\text{ideal}}$: dict mapping reconstruction types to image paths
- $\mathcal{D}_{\text{traj}}$: list of trajectory image paths

**Output:** Batch of training examples

```
Procedure SampleIdealAnchoring( D_ideal, D_traj ):
    
    // 4.1 Select a reconstruction type with ideal images
    available_types ← {t ∈ D_ideal : D_ideal[t] ≠ ∅}
    ideal_type ← UniformRandom( available_types )
    type_idx ← TYPE_TO_IDX[ideal_type]
    
    ideal_path ← UniformRandom( D_ideal[ideal_type] )
    
    // 4.2 Select second image (70% trajectory, 30% other ideal type)
    If UniformRandom(0,1) < 0.7:
        other_path ← UniformRandom( D_traj )
    Else:
        other_types ← available_types \ {ideal_type}
        other_type ← UniformRandom( other_types )
        other_path ← UniformRandom( D_ideal[other_type] )
    
    // 4.3 Load and transform
    img_ideal ← LoadAndTransform(ideal_path)
    img_other ← LoadAndTransform(other_path)
    
    // 4.4 Randomize order, ideal always wins
    If UniformRandom(0,1) < 0.5:
        Return (img_ideal, img_other, type_idx, winner=0, weight=1.0)
    Else:
        Return (img_other, img_ideal, type_idx, winner=1, weight=1.0)
```

---

## Algorithm 5: Bad Image Sampling

**Input:** 
- $\mathcal{D}_{\text{bad}}$: list of bad image paths
- $\mathcal{D}_{\text{traj}}$: list of trajectory image paths

**Output:** Batch of training examples

```
Procedure SampleBadImage( D_bad, D_traj ):
    
    bad_path ← UniformRandom( D_bad )
    traj_path ← UniformRandom( D_traj )
    
    img_bad ← LoadAndTransform(bad_path)
    img_traj ← LoadAndTransform(traj_path)
    
    // Random reconstruction type
    type_idx ← UniformRandom( {0,1,2,3,4} )
    
    // Trajectory image should win
    If UniformRandom(0,1) < 0.5:
        Return (img_bad, img_traj, type_idx, winner=1, weight=1.0)
    Else:
        Return (img_traj, img_bad, type_idx, winner=0, weight=1.0)
```

---

## Key Equations

| Equation | Description |
|----------|-------------|
| $\sigma(u) = \frac{1}{1 + e^{-u}}$ | Sigmoid function |
| $P(y_1 \succ y_2 \mid x) = \sigma(r(x, y_1) - r(x, y_2))$ | Bradley-Terry preference probability |
| $\mathcal{L}_{\text{BT}} = -\mathbb{E}[\log \sigma(r\_{\text{chosen}} - r\_{\text{rejected}})]$ | Standard Bradley-Terry loss |

---

## Model Architecture Summary

```
Input: image (224 × 224 × 1)
   │
   ▼
┌─────────────────────────────────────┐
│         ResNet-18 Encoder           │
│  (pretrained with SimCLR on RHEED)  │
│         Output: 512-dim vector      │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│           Reward Head               │
│  Linear(512 → 256) → ReLU           │
│  Dropout(0.1) → Linear(256 → 5)     │
└─────────────────────────────────────┘
   │
   ▼
Output: 5 reward scores [r₁, r₂, r₃, r₄, r₅]
```

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate $\eta$ | $1 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Batch size $B$ | 16 |
| Epochs $T$ | 30 |
| Gradient clipping | $\|\nabla_\theta \mathcal{L}\|_2 \leq 1.0$ |
| Dropout rate | 0.1 |

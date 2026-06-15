# Task 1: Algorithm Pseudo-code - Bradley-Terry Reward Model

## 1. Main Training Loop

```python
def train_model(pairwise_data, ideal_images, bad_images, trajectory_images):
    """
    Main training loop for Bradley-Terry reward model.
    
    Args:
        pairwise_data: DataFrame with 235 human preference comparisons
        ideal_images: Dict of 40 high-quality reference images per type
        bad_images: List of 13 low-quality images
        trajectory_images: List of 1,124 unlabeled images
    """
    
    # Initialize encoder and reward head
    encoder = load_pretrained_resnet18("simclr_rheed.pth")
    reward_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 5)
    )
    model = BradleyTerryModel(encoder, reward_head)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
    best_loss = float('inf')
    
    for epoch in range(1, 31):
        total_loss = 0
        
        for step in range(500):  # samples_per_epoch
            # Sample data source (60% pairwise, 25% ideal, 15% bad)
            r = random.random()
            
            if r < 0.60 and pairwise_data:
                batch = sample_pairwise(pairwise_data)
            elif r < 0.85 and ideal_images:
                batch = sample_ideal_anchoring(ideal_images, trajectory_images)
            else:
                batch = sample_bad_image(bad_images, trajectory_images)
            
            # Forward pass
            r1 = model(batch['img1'])  # [B, 5]
            r2 = model(batch['img2'])  # [B, 5]
            
            # Compute Bradley-Terry loss
            loss = compute_bt_loss(r1, r2, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / 500
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, "best_model.pth")
        
        print(f"Epoch {epoch:2d}/30 | Loss: {avg_loss:.4f}")
    
    return model
```

---

## 2. Bradley-Terry Loss Function

```python
def compute_bt_loss(r1, r2, batch):
    """
    Compute Bradley-Terry loss for preference learning.
    
    Args:
        r1: [B, 5] reward scores for image 1
        r2: [B, 5] reward scores for image 2
        batch: dict with keys:
            - type_idx: [B] reconstruction type (0-4)
            - winner: [B] 0=img1_wins, 1=img2_wins, 2=tie, 3=not_apply
            - weight: [B] confidence weight (1.0 or 0.7)
    """
    batch_size = r1.shape[0]
    type_idx = batch['type_idx']
    winner = batch['winner']
    weight = batch['weight']
    
    # Extract scores for the relevant reconstruction type
    idx = torch.arange(batch_size)
    r1_t = r1[idx, type_idx]
    r2_t = r2[idx, type_idx]
    
    loss = 0.0
    
    # Case 1: Image 1 wins -> maximize P(img1 > img2)
    mask = (winner == 0)
    if mask.any():
        l = -torch.log(torch.sigmoid(r1_t[mask] - r2_t[mask]))
        loss += (l * weight[mask]).mean() * mask.sum() / batch_size
    
    # Case 2: Image 2 wins -> maximize P(img2 > img1)
    mask = (winner == 1)
    if mask.any():
        l = -torch.log(torch.sigmoid(r2_t[mask] - r1_t[mask]))
        loss += (l * weight[mask]).mean() * mask.sum() / batch_size
    
    # Case 3: Tie -> minimize |r1 - r2|
    mask = (winner == 2)
    if mask.any():
        l = torch.abs(r1_t[mask] - r2_t[mask])
        loss += (l * weight[mask]).mean() * mask.sum() / batch_size
    
    # Case 4: Not applicable -> both scores should be low
    mask = (winner == 3)
    if mask.any():
        l = torch.relu(r1_t[mask]) + torch.relu(r2_t[mask])
        loss += (l * weight[mask]).mean() * mask.sum() / batch_size
    
    return loss
```

---

## 3. Pairwise Sampling

```python
def sample_pairwise(pairwise_df):
    """Sample a pairwise comparison with confidence weighting."""
    row = pairwise_df.sample(1).iloc[0]
    
    path1 = row['Image1_Path']
    path2 = row['Image2_Path']
    rt = row['Reconstruction_Type']      # e.g., "HTR"
    winner = str(row['Winner'])          # "1", "2", "tie", "not_apply"
    confidence = row.get('Confidence', 'Confident')
    
    # Confidence weights
    weight = 1.0 if confidence == 'Confident' else 0.7
    
    # Data augmentation: swap images with 50% probability
    if random.random() < 0.5:
        path1, path2 = path2, path1
        winner = {'1': '2', '2': '1'}.get(winner, winner)
    
    img1 = load_and_transform(path1)  # [3, 224, 224]
    img2 = load_and_transform(path2)
    
    type_idx = RECONSTRUCTION_TYPES.index(rt)
    winner_idx = {'1': 0, '2': 1, 'tie': 2, 'not_apply': 3}[winner]
    
    return {
        'img1': img1,
        'img2': img2,
        'type_idx': type_idx,
        'winner': winner_idx,
        'weight': weight
    }
```

---

## 4. Ideal Anchoring Sampling

```python
def sample_ideal_anchoring(ideal_images, trajectory_images):
    """Create ideal vs random or ideal vs other-type pairs."""
    
    # Select a reconstruction type with ideal images
    types_with_ideal = [t for t, imgs in ideal_images.items() if imgs]
    ideal_type = random.choice(types_with_ideal)
    type_idx = RECONSTRUCTION_TYPES.index(ideal_type)
    
    ideal_path = random.choice(ideal_images[ideal_type])
    
    # 70%: ideal vs random trajectory
    if random.random() < 0.7:
        other_path = random.choice(trajectory_images)
    else:
        # 30%: ideal vs other-type ideal (cross-type anchoring)
        other_types = [t for t in types_with_ideal if t != ideal_type]
        other_type = random.choice(other_types)
        other_path = random.choice(ideal_images[other_type])
    
    img_ideal = load_and_transform(ideal_path)
    img_other = load_and_transform(other_path)
    
    # Randomize order, ideal always wins
    if random.random() < 0.5:
        return {
            'img1': img_ideal,
            'img2': img_other,
            'type_idx': type_idx,
            'winner': 0,      # img1 wins
            'weight': 1.0
        }
    else:
        return {
            'img1': img_other,
            'img2': img_ideal,
            'type_idx': type_idx,
            'winner': 1,      # img2 wins
            'weight': 1.0
        }
```

---

## 5. Bad Image Sampling

```python
def sample_bad_image(bad_images, trajectory_images):
    """Bad image vs trajectory - trajectory should win."""
    
    bad_path = random.choice(bad_images)
    traj_path = random.choice(trajectory_images)
    
    img_bad = load_and_transform(bad_path)
    img_traj = load_and_transform(traj_path)
    
    # Random reconstruction type
    type_idx = random.randint(0, 4)
    
    # Trajectory wins over bad image
    if random.random() < 0.5:
        return {
            'img1': img_bad,
            'img2': img_traj,
            'type_idx': type_idx,
            'winner': 1,      # img2 (traj) wins
            'weight': 1.0
        }
    else:
        return {
            'img1': img_traj,
            'img2': img_bad,
            'type_idx': type_idx,
            'winner': 0,      # img1 (traj) wins
            'weight': 1.0
        }
```

---

## 6. Model Architecture

```python
class BradleyTerryModel(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # ResNet-18 encoder (pretrained with SimCLR)
        backbone = models.resnet18(pretrained=False)
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)  # Output: [B, 512, 1, 1]
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.encoder.load_state_dict(torch.load(pretrained_path))
        
        # Reward head: 512 -> 256 -> 5
        self.reward_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 5)   # 5 reconstruction types
        )
    
    def forward(self, x):
        # x: [B, 3, 224, 224]
        h = self.encoder(x)           # [B, 512, 1, 1]
        h = h.squeeze(-1).squeeze(-1)  # [B, 512]
        rewards = self.reward_head(h)  # [B, 5]
        return rewards
```

---

## 7. Evaluation - Pairwise Accuracy

```python
def evaluate_pairwise(model, test_df, device):
    """Evaluate on held-out pairwise comparisons."""
    model.eval()
    
    results = {rt: {'correct': 0, 'total': 0} 
               for rt in RECONSTRUCTION_TYPES}
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            rt = row['Reconstruction_Type']
            winner = str(row['Winner'])
            
            if rt not in TYPE_TO_IDX or winner not in WINNER_MAP:
                continue
            
            type_idx = TYPE_TO_IDX[rt]
            
            # Load images
            img1 = load_image(row['Image1_Path']).unsqueeze(0).to(device)
            img2 = load_image(row['Image2_Path']).unsqueeze(0).to(device)
            
            # Get scores
            r1 = model(img1)[0, type_idx].item()
            r2 = model(img2)[0, type_idx].item()
            
            # Determine correctness
            if winner == '1':
                correct = (r1 > r2)
            elif winner == '2':
                correct = (r2 > r1)
            elif winner == 'tie':
                correct = (abs(r1 - r2) < 0.5)
            elif winner == 'not_apply':
                correct = (r1 < 0 and r2 < 0)
            
            results[rt]['total'] += 1
            if correct:
                results[rt]['correct'] += 1
    
    return results
```

---

## 8. Evaluation - Single Image Classification

```python
def classify_image(model, image_path, device):
    """Classify a single image into one of 5 reconstruction types."""
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        scores = model(img_tensor).squeeze().cpu().numpy()  # [5]
    
    # Convert to probabilities via softmax
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]
    
    return {
        'predicted': RECONSTRUCTION_TYPES[predicted_class],
        'confidence': confidence,
        'scores': dict(zip(RECONSTRUCTION_TYPES, scores)),
        'probs': dict(zip(RECONSTRUCTION_TYPES, probs))
    }
```

---

## 9. Training Configuration Summary

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 16 |
| Epochs | 30 |
| LR Schedule | Cosine Annealing |
| Gradient Clipping | max_norm = 1.0 |
| Dropout | 0.1 |
| Samples per Epoch | 500 |
| Data Source Split | 60% pairwise, 25% ideal, 15% bad |

---

## 10. Key Formulas

| Component | Formula |
|-----------|---------|
| Bradley-Terry Probability | $P(A \succ B) = \sigma(r_A - r_B) = \frac{1}{1 + e^{-(r_A - r_B)}}$ |
| Preference Loss | $\mathcal{L}_{\text{pref}} = -\log\sigma(r_{\text{chosen}} - r_{\text{rejected}})$ |
| Tie Loss | $\mathcal{L}_{\text{tie}} = |r_1 - r_2|$ |
| Not-Apply Loss | $\mathcal{L}_{\text{na}} = \max(0, r_1) + \max(0, r_2)$ |

---

## 11. Implementation Notes

**Key Details:**

1. **Data Augmentation:** 50% random swapping of image pairs with corresponding winner label adjustment

2. **Confidence Weighting:** 
   - "Confident" labels: weight = 1.0
   - "Somewhat sure" labels: weight = 0.7

3. **Cross-Type Anchoring:** 30% of ideal samples compare different reconstruction types to improve class separation

4. **Evaluation Thresholds:**
   - Ties: correct if |r1 - r2| < 0.5
   - Not applicable: correct if r1 < 0 and r2 < 0

5. **Data Split:** Split at pair level (not comparison level) to prevent data leakage
   - Training: 72 pairs (185 comparisons)
   - Test: 18 pairs (48 comparisons)

6. **Model is Data-Limited:** Learning curve shows no plateau; more labels will improve performance

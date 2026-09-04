"""Resumable training for static peak-aware Bradley--Terry reward models."""
from __future__ import annotations

import json, time
from pathlib import Path
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

from .peak_aware_static_rheed_reward_model import PeakAwareStaticRHEEDRewardModel, RECONSTRUCTION_TYPES
from .pairwise_and_absolute_label_dataset import load_pairwise_rows, load_absolute_and_ideal_anchors

TYPE_INDEX = {name: index for index, name in enumerate(RECONSTRUCTION_TYPES)}
TRANSFORM = T.Compose([T.Resize((224, 224)), T.Grayscale(3), T.ToTensor(), T.Normalize((.5,)*3, (.25,)*3)])

class PairDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        load = lambda path: TRANSFORM(Image.open(path).convert("RGB"))
        return load(row["left"]), load(row["right"]), TYPE_INDEX.get(row["reconstruction_type"], 0), row["winner"]

def _loss(rewards_left, rewards_right, type_indices, winners):
    chosen, rejected, ties, not_apply = [], [], [], []
    for index, winner in enumerate(winners):
        left, right = rewards_left[index, type_indices[index]], rewards_right[index, type_indices[index]]
        if winner == "1": chosen.append(left - right)
        elif winner == "2": chosen.append(right - left)
        elif winner == "tie": ties.append((left - right).square())
        else: not_apply.extend((left.square(), right.square()))
    values = []
    if chosen: values.append(-F.logsigmoid(torch.stack(chosen)).mean())
    if ties: values.append(torch.stack(ties).mean())
    if not_apply: values.append(torch.stack(not_apply).mean())
    return torch.stack(values).mean() if values else rewards_left.sum() * 0

def train_job(data_root, split, output_directory, use_peak_features, seed, device="cpu", epochs=12, batch_size=8, resume=True):
    torch.manual_seed(seed)
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    rows = load_pairwise_rows(data_root)
    train_images = set(split["images"]["train"])
    test_images = set(split["images"]["test"])
    train_rows = [row for row in rows if row["left"] in train_images and row["right"] in train_images]
    if not train_rows: raise ValueError("Image-disjoint split left no train-only pairs; adjust the split before training.")
    loader = DataLoader(PairDataset(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
    anchors = [anchor for anchor in load_absolute_and_ideal_anchors(data_root) if anchor["path"] not in test_images]
    model = PeakAwareStaticRHEEDRewardModel(use_peak_features=use_peak_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    checkpoint = output / "resumable_training_checkpoint.pth"
    start = 0
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start = state["epoch"] + 1
    history = []
    for epoch in range(start, epochs):
        model.train(); losses = []
        for left, right, indices, winners in tqdm(loader, desc=f"seed {seed} epoch {epoch + 1}/{epochs}", leave=False):
            left, right, indices = left.to(device), right.to(device), indices.to(device)
            optimizer.zero_grad(); loss = _loss(model(left), model(right), indices, winners); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        if anchors:
            sampled = anchors[:min(len(anchors), 32)]
            images = torch.stack([TRANSFORM(Image.open(anchor["path"]).convert("RGB")) for anchor in sampled]).to(device)
            optimizer.zero_grad(); rewards = model(images); quality = model.quality_score(images)
            quality_targets = torch.tensor([anchor["quality_target"] for anchor in sampled], device=device)
            anchor_loss = F.binary_cross_entropy_with_logits(quality, quality_targets)
            labelled = [(index, anchor) for index, anchor in enumerate(sampled) if anchor["reconstruction_type"] in TYPE_INDEX]
            if labelled:
                row_indices = torch.tensor([item[0] for item in labelled], device=device)
                class_indices = torch.tensor([TYPE_INDEX[item[1]["reconstruction_type"]] for item in labelled], device=device)
                anchor_loss = anchor_loss + F.binary_cross_entropy_with_logits(rewards[row_indices, class_indices], torch.ones(len(labelled), device=device))
            anchor_loss.backward(); optimizer.step(); losses.append(float(anchor_loss.detach().cpu()))
        progress = {"status": "running", "seed": seed, "epoch": epoch + 1, "epochs": epochs, "mean_loss": sum(losses)/len(losses), "elapsed_seconds": time.time()}
        history.append(progress)
        (output / "training_progress.json").write_text(json.dumps({"history": history, "latest": progress}, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, checkpoint)
    validation_images = set(split["images"]["validation"])
    validation_rows = [row for row in rows if row["left"] in validation_images and row["right"] in validation_images and row["winner"] in {"1", "2"}]
    model.eval(); correct = 0
    with torch.no_grad():
        for row in validation_rows:
            left = TRANSFORM(Image.open(row["left"]).convert("RGB")).unsqueeze(0).to(device)
            right = TRANSFORM(Image.open(row["right"]).convert("RGB")).unsqueeze(0).to(device)
            probability = float(model.pairwise_probability(left, right, TYPE_INDEX.get(row["reconstruction_type"], 0)).cpu())
            correct += ("1" if probability > .5 else "2") == row["winner"]
    final_weights = output / "final_model_weights.pth"; torch.save(model.state_dict(), final_weights)
    final = {"status": "completed", "seed": seed, "variant": "image_encoder_plus_peak_features" if use_peak_features else "image_encoder_only", "final_model_weights": str(final_weights), "train_pairs": len(train_rows), "direct_label_anchor_count": len(anchors), "validation_decisive_pairs": len(validation_rows), "validation_pairwise_winner_accuracy": correct / len(validation_rows) if validation_rows else None, "epochs": epochs}
    (output / "completed_task_result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    return final

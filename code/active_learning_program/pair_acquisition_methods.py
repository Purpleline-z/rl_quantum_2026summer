"""Comparable, streamed candidate-pair selection strategies."""

from __future__ import annotations

import gc
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
from image_metadata_and_symmetry_features import symmetry_image

BATCH_SIZE = 32


def _rewards(model, embeddings, metadata=None):
    """Score cached image embeddings while preserving optional metadata fusion."""
    if getattr(model, "metadata_dim", 0):
        if metadata is None:
            raise ValueError("Metadata-fusion model requires candidate metadata vectors.")
        return model.reward_from_embeddings(embeddings, metadata)
    return model.reward_head(embeddings)


class _CandidateDataset(Dataset):
    def __init__(self, candidates, symmetry_mode="none"):
        self.candidates = candidates
        self.symmetry_mode = symmetry_mode
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                    T.Normalize(mean=[.5] * 3, std=[.25] * 3)])

    def __len__(self): return len(self.candidates)

    def __getitem__(self, index):
        pair = self.candidates[index]
        with Image.open(pair["img1"]) as left, Image.open(pair["img2"]) as right:
            return self.transform(symmetry_image(left, self.symmetry_mode)), self.transform(symmetry_image(right, self.symmetry_mode)), index


class _ImageDataset(Dataset):
    def __init__(self, paths, symmetry_mode="none"):
        self.paths = paths
        self.symmetry_mode = symmetry_mode
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                                    T.Normalize(mean=[.5] * 3, std=[.25] * 3)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(symmetry_image(image, self.symmetry_mode)), path


def build_embedding_cache(candidates, model, device, symmetry_mode="none") -> dict[str, torch.Tensor]:
    """Encode every unique candidate image once; CPU cache is ~2 KiB/image."""
    paths = sorted({pair["img1"] for pair in candidates} | {pair["img2"] for pair in candidates})
    loader = DataLoader(_ImageDataset(paths, symmetry_mode), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=torch.device(device).type == "cuda")
    cache = {}; model = model.to(device).eval()
    with torch.no_grad():
        for images, batch_paths in loader:
            features = model.encoder(images.to(device)).cpu()
            cache.update(zip(batch_paths, features))
            del images, features
    del loader
    gc.collect()
    return cache


def score_uncertainty(candidates, model, device, embedding_cache=None, symmetry_mode="none") -> list[dict]:
    """Mean Bernoulli entropy across all Bradley--Terry heads; streamed in batches of 32."""
    model = model.to(device).eval()
    if embedding_cache is not None:
        scored = []
        with torch.no_grad():
            for start in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[start:start + BATCH_SIZE]
                left = torch.stack([embedding_cache[pair["img1"]] for pair in batch]).to(device)
                right = torch.stack([embedding_cache[pair["img2"]] for pair in batch]).to(device)
                left_metadata = right_metadata = None
                if getattr(model, "metadata_dim", 0):
                    left_metadata = torch.tensor([pair["metadata1"] for pair in batch], dtype=torch.float32, device=device)
                    right_metadata = torch.tensor([pair["metadata2"] for pair in batch], dtype=torch.float32, device=device)
                probabilities = torch.sigmoid(_rewards(model, left, left_metadata) - _rewards(model, right, right_metadata)).clamp(1e-7, 1 - 1e-7)
                entropy = -(probabilities * probabilities.log() + (1 - probabilities) * (1 - probabilities).log()).mean(1)
                scored.extend({**pair, "uncertainty": float(value)} for pair, value in zip(batch, entropy.cpu().tolist()))
                del left, right, probabilities, entropy
        return scored
    loader = DataLoader(_CandidateDataset(candidates, symmetry_mode), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=torch.device(device).type == "cuda")
    scored = []
    with torch.no_grad():
        for left, right, indices in loader:
            left, right = left.to(device), right.to(device)
            probabilities = torch.sigmoid(model(left) - model(right)).clamp(1e-7, 1 - 1e-7)
            entropy = -(probabilities * probabilities.log() + (1 - probabilities) * (1 - probabilities).log()).mean(1)
            for index, value in zip(indices.tolist(), entropy.cpu().tolist()):
                scored.append({**candidates[index], "uncertainty": float(value)})
            del left, right, probabilities, entropy
    del loader
    gc.collect()
    return scored


def random_sampling(candidates, current_model, budget, device="cpu", seed=42, embedding_cache=None) -> list[dict]:
    """Random baseline: choose K pairs uniformly without replacement."""
    del current_model, device, embedding_cache
    return random.Random(seed).sample(candidates, min(budget, len(candidates)))


def uncertainty_sampling(candidates, current_model, budget, device="cpu", seed=42, embedding_cache=None) -> list[dict]:
    """Pure uncertainty baseline: choose the K largest multi-head entropies."""
    del seed
    return sorted(score_uncertainty(candidates, current_model, device, embedding_cache), key=lambda pair: pair["uncertainty"], reverse=True)[:budget]


def cluster_quota_uncertainty_sampling(candidates, current_model, budget, device="cpu", seed=42, embedding_cache=None) -> list[dict]:
    """Custom cluster-quota uncertainty heuristic; not Cluster-Margin."""
    del seed
    grouped = defaultdict(list)
    for pair in score_uncertainty(candidates, current_model, device, embedding_cache):
        grouped[int(pair["cluster1"])].append(pair)
    total = sum(map(len, grouped.values()))
    selected, used = [], set()
    for cluster_id, pairs in sorted(grouped.items()):
        quota = max(1, round(budget * len(pairs) / max(1, total)))
        for pair in sorted(pairs, key=lambda item: item["uncertainty"], reverse=True)[:quota]:
            key = tuple(sorted((pair["img1"], pair["img2"])))
            if key not in used and len(selected) < budget:
                selected.append(pair); used.add(key)
    remaining = sorted((pair for pairs in grouped.values() for pair in pairs), key=lambda item: item["uncertainty"], reverse=True)
    for pair in remaining:
        if len(selected) >= budget: break
        key = tuple(sorted((pair["img1"], pair["img2"])))
        if key not in used:
            selected.append(pair); used.add(key)
    return selected


def cluster_margin_pairwise_sampling(candidates, current_model, budget, device="cpu", seed=42, embedding_cache=None) -> list[dict]:
    """Low-budget Cluster-Margin adaptation with auditable round-robin selection."""
    del seed
    scored = score_uncertainty(candidates, current_model, device, embedding_cache)
    # Entropy is retained for comparability; margin drives this selector.
    if embedding_cache is None:
        raise ValueError("cluster_margin_pairwise_sampling requires an embedding cache.")
    margins = []
    current_model.eval()
    with torch.no_grad():
        for pair in scored:
            left = embedding_cache[pair["img1"]].unsqueeze(0).to(device)
            right = embedding_cache[pair["img2"]].unsqueeze(0).to(device)
            ma = mb = None
            if getattr(current_model, "metadata_dim", 0):
                ma = torch.tensor(pair["metadata1"], dtype=torch.float32, device=device).unsqueeze(0)
                mb = torch.tensor(pair["metadata2"], dtype=torch.float32, device=device).unsqueeze(0)
            probability = torch.sigmoid(_rewards(current_model, left, ma) - _rewards(current_model, right, mb))
            margins.append(float(torch.abs(probability - .5).mean().cpu()))
    for pair, margin in zip(scored, margins): pair["margin"] = margin
    prefilter_size = min(max(1, 10 * budget), len(scored))
    prefiltered = sorted(scored, key=lambda item: item["margin"])[:prefilter_size]
    full_cluster_sizes = defaultdict(int)
    for pair in scored:
        full_cluster_sizes[int(pair["cluster1"])] += 1
    grouped = defaultdict(list)
    for pair in prefiltered: grouped[int(pair["cluster1"])].append(pair)
    for cluster, members in grouped.items():
        members.sort(key=lambda item: item["margin"])
        for member in members:
            member["prefilter_cluster_size"] = len(members); member["prefilter_size"] = prefilter_size
    selected, rank = [], 1
    while len(selected) < min(budget, len(prefiltered)):
        progress = False
        for cluster, members in sorted(grouped.items(), key=lambda item: (len(item[1]), item[0])):
            if not members or len(selected) >= budget: continue
            item = members.pop(0); item["selection_rank"] = rank; rank += 1
            selected.append(item); progress = True
        if not progress: break
    # Preserve a complete candidate-level audit on the original list so the
    # controlled runner can export every margin, cluster, and prefilter flag.
    selected_ranks = {item["pair_id"]: item["selection_rank"] for item in selected}
    prefiltered_ids = {item["pair_id"] for item in prefiltered}
    scored_by_id = {item["pair_id"]: item for item in scored}
    for item in candidates:
        scored_item = scored_by_id[item["pair_id"]]
        cluster = int(scored_item["cluster1"])
        item.update({"cluster_margin": float(scored_item["margin"]), "cluster_size": int(full_cluster_sizes[cluster]),
                     "prefilter_member": item["pair_id"] in prefiltered_ids, "prefilter_size": prefilter_size,
                     "prefilter_cluster_size": int(sum(1 for x in prefiltered if int(x["cluster1"]) == cluster)),
                     "selection_rank": selected_ranks.get(item["pair_id"])})
    return selected

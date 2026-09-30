from __future__ import annotations

import json
import os
import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau
from tqdm import tqdm

K = 5
D = 32
DEFAULT_BUDGETS = [10, 20, 40, 80, 160, 320]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def _generate_ground_truth(n_images: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((K, n_images))


def _generate_features(n_images: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.standard_normal((n_images, D))
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    return phi / norms


def _build_candidate_pool(
    n_images: int, ground_truth: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_i, idx_j = zip(*combinations(range(n_images), 2))
    idx_i = np.array(idx_i, dtype=np.int32)
    idx_j = np.array(idx_j, dtype=np.int32)
    type_idx = rng.integers(0, K, size=len(idx_i), dtype=np.int32)
    p_win = _sigmoid(ground_truth[type_idx, idx_i] - ground_truth[type_idx, idx_j])
    labels = rng.binomial(1, p_win).astype(np.int32)
    return idx_i, idx_j, type_idx, labels


def _train_split(
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    type_idx: np.ndarray,
    labels: np.ndarray,
    test_frac: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(idx_i)
    perm = rng.permutation(n) if rng is not None else np.random.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_mask = np.zeros(n, dtype=bool)
    test_mask[perm[:n_test]] = True
    train_mask = ~test_mask
    return train_mask, test_mask


class MultiHeadBTModel(nn.Module):
    def __init__(self, d: int = D, k: int = K):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(k, d))

    def forward(self, phi_i: torch.Tensor, phi_j: torch.Tensor) -> torch.Tensor:
        r_i = (self.weights @ phi_i.T)
        r_j = (self.weights @ phi_j.T)
        return r_i - r_j

    def rewards(self, phi: torch.Tensor) -> torch.Tensor:
        return (self.weights @ phi.T).T


def _train_model(
    model: MultiHeadBTModel,
    phi: np.ndarray,
    labeled_indices: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    type_idx: np.ndarray,
    labels: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
) -> None:
    if len(labeled_indices) == 0:
        return

    phi_t = torch.tensor(phi, dtype=torch.float32)
    li = idx_i[labeled_indices]
    lj = idx_j[labeled_indices]
    lt = type_idx[labeled_indices]
    ll = labels[labeled_indices]

    phi_i = phi_t[li]
    phi_j = phi_t[lj]
    type_t = torch.tensor(lt, dtype=torch.long)
    label_t = torch.tensor(ll, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        diff = model(phi_i, phi_j)
        row_idx = torch.arange(len(type_t))
        logit = diff[type_t, row_idx]
        loss = nn.functional.binary_cross_entropy_with_logits(logit, label_t)
        loss.backward()
        optimizer.step()


def _evaluate(
    model: MultiHeadBTModel,
    phi: np.ndarray,
    ground_truth: np.ndarray,
    test_indices: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    type_idx: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    phi_t = torch.tensor(phi, dtype=torch.float32)
    with torch.no_grad():
        pred_rewards = model.rewards(phi_t).numpy()

    taus = []
    for k in range(K):
        tau, _ = kendalltau(pred_rewards[:, k], ground_truth[k])
        taus.append(tau)
    mean_tau = float(np.mean(taus))

    if len(test_indices) == 0:
        return mean_tau, 0.0

    ti = idx_i[test_indices]
    tj = idx_j[test_indices]
    tt = type_idx[test_indices]
    tl = labels[test_indices]

    pred_diff = pred_rewards[ti, tt] - pred_rewards[tj, tt]
    pred_label = (pred_diff > 0).astype(np.int32)
    accuracy = float(np.mean(pred_label == tl))
    return mean_tau, accuracy


def _acquire_random(
    candidate_indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    chosen = rng.choice(candidate_indices, size=min(n, len(candidate_indices)), replace=False)
    return chosen


def _acquire_uncertainty_type_aware(
    candidate_indices: np.ndarray,
    n: int,
    model: MultiHeadBTModel,
    phi: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    type_idx: np.ndarray,
    **kwargs,
) -> np.ndarray:
    w = model.weights.detach().numpy()
    ci = idx_i[candidate_indices]
    cj = idx_j[candidate_indices]
    ct = type_idx[candidate_indices]
    diff = np.einsum("kd,kd->k", w[ct], phi[ci] - phi[cj])
    p = _sigmoid(diff)
    scores = _entropy(p)
    top = np.argsort(-scores)[:n]
    return candidate_indices[top]


def _acquire_uncertainty_avg(
    candidate_indices: np.ndarray,
    n: int,
    model: MultiHeadBTModel,
    phi: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    **kwargs,
) -> np.ndarray:
    w = model.weights.detach().numpy()
    ci = idx_i[candidate_indices]
    cj = idx_j[candidate_indices]
    diff_all = (w @ (phi[ci] - phi[cj]).T)
    p_all = _sigmoid(diff_all)
    ent_all = _entropy(p_all)
    scores = ent_all.mean(axis=0)
    top = np.argsort(-scores)[:n]
    return candidate_indices[top]


def _acquire_coreset_greedy(
    candidate_indices: np.ndarray,
    n: int,
    phi: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    already_selected: np.ndarray,
    **kwargs,
) -> np.ndarray:
    ci = idx_i[candidate_indices]
    cj = idx_j[candidate_indices]
    pair_feats = np.concatenate([phi[ci], phi[cj]], axis=1)

    if len(already_selected) > 0:
        si = idx_i[already_selected]
        sj = idx_j[already_selected]
        sel_feats = np.concatenate([phi[si], phi[sj]], axis=1)
        dists = np.min(
            np.sum((pair_feats[:, None, :] - sel_feats[None, :, :]) ** 2, axis=2), axis=1
        )
    else:
        dists = np.full(len(candidate_indices), np.inf)

    chosen = []
    current_dists = dists.copy()

    for _ in range(min(n, len(candidate_indices))):
        pick = int(np.argmax(current_dists))
        chosen.append(candidate_indices[pick])
        new_d = np.sum((pair_feats - pair_feats[pick]) ** 2, axis=1)
        current_dists = np.minimum(current_dists, new_d)

    return np.array(chosen)


def _acquire_mc_bald_type_aware(
    candidate_indices: np.ndarray,
    n: int,
    model: MultiHeadBTModel,
    phi: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    type_idx: np.ndarray,
    n_mc: int = 20,
    sigma: float = 0.1,
    **kwargs,
) -> np.ndarray:
    w_base = model.weights.detach().numpy()
    ci = idx_i[candidate_indices]
    cj = idx_j[candidate_indices]
    ct = type_idx[candidate_indices]
    delta = phi[ci] - phi[cj]

    rng_mc = np.random.default_rng(42)
    all_probs = []
    for _ in range(n_mc):
        w_noisy = w_base + rng_mc.normal(0, sigma, w_base.shape)
        diff = np.einsum("kd,kd->k", w_noisy[ct], delta)
        all_probs.append(_sigmoid(diff))

    all_probs = np.stack(all_probs, axis=0)
    p_mean = all_probs.mean(axis=0)
    H_mean = _entropy(p_mean)
    mean_H = _entropy(all_probs).mean(axis=0)
    bald = H_mean - mean_H
    top = np.argsort(-bald)[:n]
    return candidate_indices[top]


def _acquire_fisher_information(
    candidate_indices: np.ndarray,
    n: int,
    model: MultiHeadBTModel,
    phi: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    type_idx: np.ndarray,
    labeled_indices: np.ndarray,
    **kwargs,
) -> np.ndarray:
    w = model.weights.detach().numpy()

    fisher_per_head = []
    for k in range(K):
        F = np.eye(D) * 1e-4
        if len(labeled_indices) > 0:
            mask = type_idx[labeled_indices] == k
            li = labeled_indices[mask]
            if len(li) > 0:
                ci = idx_i[li]
                cj = idx_j[li]
                delta = phi[ci] - phi[cj]
                diff = delta @ w[k]
                p = _sigmoid(diff)
                weight = p * (1 - p)
                F = F + (delta.T * weight) @ delta
        fisher_per_head.append(F)

    ci = idx_i[candidate_indices]
    cj = idx_j[candidate_indices]
    ct = type_idx[candidate_indices]
    delta = phi[ci] - phi[cj]

    scores = np.zeros(len(candidate_indices))
    for idx_in_cand in range(len(candidate_indices)):
        k = ct[idx_in_cand]
        d = delta[idx_in_cand]
        F = fisher_per_head[k]
        diff_val = float(d @ w[k])
        p = float(_sigmoid(diff_val))
        weight = p * (1 - p)
        Fd = F @ d
        gain = weight * (d @ Fd) / (1.0 + weight * (d @ Fd) + 1e-12)
        scores[idx_in_cand] = gain

    top = np.argsort(-scores)[:n]
    return candidate_indices[top]


STRATEGIES = {
    "random": _acquire_random,
    "uncertainty_type_aware": _acquire_uncertainty_type_aware,
    "uncertainty_avg": _acquire_uncertainty_avg,
    "coreset_greedy": _acquire_coreset_greedy,
    "mc_bald_type_aware": _acquire_mc_bald_type_aware,
    "fisher_information": _acquire_fisher_information,
}


def run_benchmark(
    n_images: int = 200,
    seed: int = 0,
    budgets: Optional[List[int]] = None,
) -> Dict[str, List[Tuple[int, float, float]]]:
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    rng = np.random.default_rng(seed)
    ground_truth = _generate_ground_truth(n_images, rng)
    phi = _generate_features(n_images, rng)
    idx_i, idx_j, type_idx, labels = _build_candidate_pool(n_images, ground_truth, rng)

    n_pairs = len(idx_i)
    all_indices = np.arange(n_pairs)
    train_mask, test_mask = _train_split(idx_i, idx_j, type_idx, labels, test_frac=0.2, rng=rng)
    trainable_pool = all_indices[train_mask]
    test_indices = all_indices[test_mask]

    results = {name: [] for name in STRATEGIES}

    for strategy_name, strategy_fn in STRATEGIES.items():
        model = MultiHeadBTModel(d=D, k=K)
        labeled_set = np.array([], dtype=np.int32)
        remaining = trainable_pool.copy()

        prev_budget = 0
        for budget in budgets:
            n_new = budget - prev_budget
            if n_new <= 0 or len(remaining) == 0:
                break

            acquire_kwargs = dict(
                candidate_indices=remaining,
                n=n_new,
                rng=rng,
                model=model,
                phi=phi,
                idx_i=idx_i,
                idx_j=idx_j,
                type_idx=type_idx,
                labels=labels,
                already_selected=labeled_set,
                labeled_indices=labeled_set,
            )

            chosen = strategy_fn(**acquire_kwargs)
            labeled_set = np.concatenate([labeled_set, chosen])
            remaining = np.setdiff1d(remaining, chosen)

            _train_model(model, phi, labeled_set, idx_i, idx_j, type_idx, labels)

            tau, acc = _evaluate(
                model, phi, ground_truth, test_indices, idx_i, idx_j, type_idx, labels
            )
            results[strategy_name].append((budget, tau, acc))
            prev_budget = budget

    return results


def run_multiple_seeds(
    n_seeds: int = 20,
    n_images: int = 200,
    budgets: Optional[List[int]] = None,
) -> Dict[str, Dict]:
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    all_results = {name: [] for name in STRATEGIES}

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        trial = run_benchmark(n_images=n_images, seed=seed, budgets=budgets)
        for name, curve in trial.items():
            all_results[name].append(curve)

    summary = {}
    for name in STRATEGIES:
        curves = all_results[name]
        n_steps = min(len(c) for c in curves)
        budgets_out = [curves[0][s][0] for s in range(n_steps)]
        taus = np.array([[c[s][1] for s in range(n_steps)] for c in curves])
        accs = np.array([[c[s][2] for s in range(n_steps)] for c in curves])
        summary[name] = {
            "budgets": budgets_out,
            "kendall_tau_mean": taus.mean(axis=0).tolist(),
            "kendall_tau_std": taus.std(axis=0).tolist(),
            "accuracy_mean": accs.mean(axis=0).tolist(),
            "accuracy_std": accs.std(axis=0).tolist(),
        }

    return summary


def _print_summary_table(summary: Dict[str, Dict]) -> None:
    strategy_names = list(summary.keys())
    budgets = summary[strategy_names[0]]["budgets"]

    header = f"{'Strategy':<30}" + "".join(f"  B={b:<6}" for b in budgets)
    print("\n" + "=" * len(header))
    print("Kendall-tau (mean ± std)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in strategy_names:
        s = summary[name]
        row = f"{name:<30}"
        for i in range(len(budgets)):
            mu = s["kendall_tau_mean"][i]
            sd = s["kendall_tau_std"][i]
            row += f"  {mu:.3f}±{sd:.3f}"
        print(row)

    print("\n" + "=" * len(header))
    print("Accuracy (mean ± std)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in strategy_names:
        s = summary[name]
        row = f"{name:<30}"
        for i in range(len(budgets)):
            mu = s["accuracy_mean"][i]
            sd = s["accuracy_std"][i]
            row += f"  {mu:.3f}±{sd:.3f}"
        print(row)

    print("=" * len(header))


if __name__ == "__main__":
    t0 = time.time()
    summary = run_multiple_seeds(n_seeds=20, n_images=200)
    _print_summary_table(summary)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

#!/usr/bin/env python3
"""Auditable single-shot performance-versus-budget experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pairwise_active_learning_pipeline import Config, Experiment, transform
from image_metadata_and_symmetry_features import normalized, symmetry_image
from project_file_locations import RESULT_ROOT
from image_mixture_features import fit_reference_nmf, score_nmf_features
from pair_acquisition_methods import score_uncertainty

ROOT = Path(__file__).resolve().parent.parent
OUT = RESULT_ROOT / "budget_curve_study"
BUDGETS, SEEDS = (10, 25, 50, 75, 100), (42, 79, 123, 202, 303)
PRIMARY = ("random", "uncertainty")
EXPLORATORY = ("cluster_quota_uncertainty", "cluster_margin_pairwise")
# Runnable selector names.  NMF strategies remain listed because their own
# diagnostic intentionally rejects acquisition until it passes.
MC_DROPOUT_STRATEGIES = ("mc_dropout_probability_variance", "mc_dropout_mutual_information", "mc_dropout_reward_variance")
ALL_STRATEGIES = PRIMARY + ("uncertainty_diversity", "core_set") + EXPLORATORY + MC_DROPOUT_STRATEGIES + ("mixture_only", "uncertainty_plus_mixture")
DATA_ROOT: str | None = None


def atomic_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def job_id(spec: dict) -> str:
    return f"seed-{spec['seed']}_budget-{spec['budget']}_strategy-{spec['strategy']}_symmetry-{spec['symmetry_mode']}"


def parse_csv_values(raw: str | None, cast, name: str) -> tuple | None:
    """Parse a compact CLI list without silently accepting malformed studies."""
    if raw is None:
        return None
    try:
        values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(f"--{name} must be a comma-separated list of valid values.") from exc
    if not values or len(set(values)) != len(values):
        raise ValueError(f"--{name} must contain one or more unique values.")
    return values


def generate_manifest(symmetry: bool = False, smoke: bool = False, exploratory: bool = False,
                      budgets: tuple[int, ...] | None = None, seeds: tuple[int, ...] | None = None,
                      strategies: tuple[str, ...] | None = None, manifest_name: str | None = None) -> Path:
    if smoke and any(value is not None for value in (budgets, seeds, strategies)):
        raise ValueError("--smoke cannot be combined with explicit budgets, seeds, or strategies.")
    budgets = (2,) if smoke else (budgets or ((100,) if symmetry else BUDGETS))
    seeds = (42, 79) if smoke else (seeds or SEEDS)
    strategies = strategies or (EXPLORATORY + ("mixture_only", "uncertainty_plus_mixture") if exploratory else PRIMARY)
    if any(budget <= 0 or budget > 120 for budget in budgets):
        raise ValueError("Each acquisition budget must be between 1 and 120 candidate pair groups.")
    unknown = sorted(set(strategies) - set(ALL_STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}. Valid values: {', '.join(ALL_STRATEGIES)}.")
    modes = ("none", "left_half_mirror", "symmetric_average") if symmetry else ("none",)
    jobs = []
    for seed in seeds:
        for budget in budgets:
            for strategy in strategies:
                for symmetry_mode in modes:
                    spec = {"seed": seed, "budget": budget, "strategy": strategy, "symmetry_mode": symmetry_mode,
                            "epoch": 3, "lr": 1e-4, "initial_pairs": 50, "candidate_pairs": 120,
                            "acquisition_mode": "single-shot"}
                    if smoke:
                        # This verifies every artifact contract quickly; it is not
                        # evidence for the fixed-protocol scientific comparison.
                        spec.update(epoch=1, initial_pairs=8, candidate_pairs=8)
                    spec["job_id"] = job_id(spec); jobs.append(spec)
    default_name = "smoke_manifest" if smoke else "symmetry_manifest" if symmetry else "exploratory_manifest" if exploratory else "budget_manifest"
    filename = manifest_name or default_name
    if not filename.endswith(".json"):
        filename += ".json"
    path = OUT / filename
    atomic_json({"study": {"budgets": budgets, "seeds": seeds, "strategies": strategies, "symmetry": symmetry,
                            "purpose": "manifest only; results are evidence only after completed jobs are aggregated"}, "jobs": jobs}, path)
    return path


def load_job(manifest: Path, requested: str) -> dict:
    matches = [x for x in json.loads(manifest.read_text(encoding="utf-8"))["jobs"] if x["job_id"] == requested or x["job_id"].startswith(requested)]
    if len(matches) != 1: raise ValueError(f"Expected one job for {requested!r}, found {len(matches)}.")
    return matches[0]


def _images(ids, exp: Experiment) -> set[str]:
    return {p for pair_id in ids for p in exp.groups[pair_id].resolved_img1.tolist() + exp.groups[pair_id].resolved_img2.tolist()}


def audit(exp: Experiment, initial: list[str], pool: list[str], selected: list[str]) -> dict:
    test = {str(x) for paths in exp.test_images.values() for x in paths}; refs = {str(x) for paths in exp.references.values() for x in paths}
    initial_images, pool_images, selected_images = _images(initial, exp), _images(pool, exp), _images(selected, exp)
    return {"source_pairwise_csv": str(exp.pairwise_csv), "source_pairwise_sha256": sha256(exp.pairwise_csv),
            "simclr_checkpoint": str(exp.encoder_weights), "simclr_checkpoint_sha256": sha256(exp.encoder_weights),
            "valid_pair_rows": int(sum(len(x) for x in exp.groups.values())), "unique_unordered_pairs": len(exp.groups),
            "type_counts": exp.rows_for(exp.groups.keys()).canonical_type.value_counts().to_dict(),
            "initial_pair_count": len(initial), "candidate_pair_count": len(pool), "selected_pair_count": len(selected),
            "initial_revealed_row_count": int(sum(len(exp.groups[x]) for x in initial)),
            "candidate_revealed_row_count": int(sum(len(exp.groups[x]) for x in pool)),
            "selected_revealed_row_count": int(sum(len(exp.groups[x]) for x in selected)),
            "exact_pair_overlap_initial_pool": len(set(initial) & set(pool)), "exact_pair_overlap_initial_selected": len(set(initial) & set(selected)),
            "image_overlap_initial_pool": len(initial_images & pool_images), "image_overlap_initial_selected": len(initial_images & selected_images),
            "pairwise_image_overlap_ideal_test": len((initial_images | pool_images) & test), "reference_test_overlap": len(refs & test),
            "ideal_test_count": len(test), "ideal_reference_count": len(refs),
            "pair_disjoint_but_not_necessarily_image_disjoint": True}


@torch.no_grad()
def _features(exp: Experiment, model, paths: list[str]) -> np.ndarray:
    tensors = []
    for path in paths:
        from PIL import Image
        with Image.open(path) as image: tensors.append(transform(exp.cfg.symmetry_mode)(symmetry_image(image, exp.cfg.symmetry_mode)))
    if not tensors: return np.empty((0, 512))
    return model.encoder(torch.stack(tensors).to(exp.device)).cpu().numpy()


def attach_feature_nmf(exp: Experiment, model, candidates: list[dict], output: Path) -> bool:
    reference_paths, labels = [], []
    for label, paths in exp.references.items():
        if label == "Twinned(2 x 1)": continue
        reference_paths.extend(map(str, paths)); labels.extend([label] * len(paths))
    diagnostic, weights = fit_reference_nmf(_features(exp, model, reference_paths), labels, n_components=4, seed=exp.cfg.seed)
    candidate_paths = [x["img1"] for x in candidates] + [x["img2"] for x in candidates]
    scores = score_nmf_features(diagnostic, _features(exp, model, candidate_paths))
    n = len(candidates)
    rows = []
    for i, item in enumerate(candidates):
        entropy = float(np.mean((scores["mixture_entropy"][i], scores["mixture_entropy"][i + n])))
        residual = float(np.mean((scores["off_basis_residual"][i], scores["off_basis_residual"][i + n])))
        item.update(feature_mixture_entropy=entropy, off_basis_residual=residual, mixture_score=entropy)
        rows.append({"pair_id": item["pair_id"], "mixture_entropy": entropy, "off_basis_residual": residual, "mixture_score": entropy})
    atomic_json({"component_class": diagnostic.component_class, "one_to_one_class_coverage": diagnostic.one_to_one,
                 "silhouette": diagnostic.silhouette, "enabled_for_acquisition": False}, output / "nmf_diagnostic.json")
    pd.DataFrame(rows).to_csv(output / "nmf_candidate_scores.csv", index=False)
    return diagnostic.one_to_one and (diagnostic.silhouette is None or diagnostic.silhouette > 0)


def select_feature_nmf(strategy: str, candidates: list[dict], model, exp: Experiment, cache: dict, budget: int, enabled: bool) -> list[dict]:
    """Exploratory NMF selectors; disabled unless the reference-only audit passes."""
    if not enabled:
        raise ValueError("Feature-space NMF reference audit did not pass; mixture-driven acquisition remains disabled.")
    n = min(budget, len(candidates))
    if strategy == "mixture_only":
        return sorted(candidates, key=lambda x: x["mixture_score"], reverse=True)[:n]
    uncertain = score_uncertainty(candidates, model, exp.device, cache)
    by_id = {x["pair_id"]: x["uncertainty"] for x in uncertain}
    u = normalized([by_id[x["pair_id"]] for x in candidates])
    m = normalized([x["mixture_score"] for x in candidates])
    scored = [{**item, "uncertainty": float(u_i), "combined_score": float((1 - exp.cfg.mixture_weight) * u_i + exp.cfg.mixture_weight * m_i)} for item, u_i, m_i in zip(candidates, u, m)]
    return sorted(scored, key=lambda x: x["combined_score"], reverse=True)[:n]


def run_job(spec: dict):
    directory = OUT / "jobs" / spec["job_id"]
    if (directory / "result.json").exists():
        print(f"{spec['job_id']}: completed"); return
    cfg = Config(initial_pairs=spec["initial_pairs"], candidate_pairs=spec["candidate_pairs"], budget=spec["budget"],
                 batch_size=5, epochs=spec["epoch"], lr=spec["lr"], seed=spec["seed"], strategies=spec["strategy"],
                 acquisition_mode="single-shot", utility_per_pair=False, symmetry_mode=spec["symmetry_mode"],
                 diversity_lambda=float(spec.get("diversity_lambda", .5)), dropout_p=float(spec.get("dropout_p", .2)),
                 mc_samples=int(spec.get("mc_samples", 20)), manifest_dir=str(directory / "manifests"), data_root=DATA_ROOT)
    exp = Experiment(cfg); exp.output = directory; exp.utility_cache_path = directory / "utility_cache.json"; exp.utility_cache = {}
    initial, pool = exp.load_and_split(); model, pre_metrics = exp.train(initial); pre = exp.evaluate(model)
    candidates, cache = exp.candidates_with_clusters(pool, model)
    nmf_usable = attach_feature_nmf(exp, model, candidates, directory) if spec.get("run_nmf_diagnostic", True) else False
    if spec["strategy"] in {"mixture_only", "uncertainty_plus_mixture"}:
        selected = select_feature_nmf(spec["strategy"], candidates, model, exp, cache, spec["budget"], nmf_usable)
    else:
        selected, selection_scores, selection_elapsed = exp.select(spec["strategy"], candidates, model, cache, [], budget=spec["budget"], labeled_ids=initial)
    if spec["strategy"] in MC_DROPOUT_STRATEGIES:
        pd.DataFrame(selection_scores).to_csv(directory / "mc_dropout_candidate_scores.csv", index=False)
    if spec["strategy"] == "cluster_margin_pairwise":
        pd.DataFrame([{key: item.get(key) for key in ("pair_id", "cluster1", "cluster_margin", "cluster_size", "prefilter_member", "prefilter_size", "prefilter_cluster_size", "selection_rank")} for item in candidates]).to_csv(directory / "cluster_margin_candidate_audit.csv", index=False)
    selected_ids = [x["pair_id"] for x in selected]
    post_model, post_metrics = exp.train(initial + selected_ids); post = exp.evaluate(post_model)
    selected_rows = exp.rows_for(selected_ids)
    selected_rows.to_csv(directory / "manifests" / "selected_pairs.csv", index=False)
    atomic_json({"spec": spec, "config": asdict(cfg), "audit": audit(exp, initial, pool, selected_ids),
                 "selected_pair_ids": selected_ids, "selected_details": selected, "pre": pre, "post": post,
                 "pre_metrics": pre_metrics, "post_metrics": post_metrics,
                 "batch_utility": post["test_accuracy"] - pre["test_accuracy"], "nmf_usable_diagnostic": nmf_usable,
                 "selection_scoring_seconds": selection_elapsed if spec["strategy"] in MC_DROPOUT_STRATEGIES else None,
                 "mc_samples": cfg.mc_samples if spec["strategy"] in MC_DROPOUT_STRATEGIES else None,
                 "dropout_p": cfg.dropout_p if spec["strategy"] in MC_DROPOUT_STRATEGIES else None}, directory / "result.json")
    print(f"{spec['job_id']}: completed")


def aggregate():
    rows = []
    for path in (OUT / "jobs").glob("*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8")); spec = result["spec"]
        rows.append({**spec, "pre_test_accuracy": result["pre"]["test_accuracy"], "post_test_accuracy": result["post"]["test_accuracy"], "batch_utility": result["batch_utility"]})
    if not rows: raise ValueError("No completed revision jobs.")
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "per_seed_results.csv", index=False)
    summary = frame.groupby(["budget", "strategy", "symmetry_mode"], as_index=False).agg(
        n=("seed", "count"), post_test_accuracy_mean=("post_test_accuracy", "mean"), post_test_accuracy_std=("post_test_accuracy", "std"),
        batch_utility_mean=("batch_utility", "mean"), batch_utility_std=("batch_utility", "std"))
    summary.to_csv(OUT / "budget_summary.csv", index=False)
    for metric, label, name in [("post_test_accuracy", "Post-acquisition fixed ideal-test accuracy", "performance_vs_budget.png"), ("batch_utility", "Batch utility", "utility_vs_budget.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (strategy, mode), group in frame.groupby(["strategy", "symmetry_mode"]):
            for seed, seed_rows in group.groupby("seed"):
                ax.plot(seed_rows.budget, seed_rows[metric], color="0.7", linewidth=.8, alpha=.7)
            stats = group.groupby("budget")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats.budget, stats["mean"], yerr=stats["std"], marker="o", capsize=3, label=f"{strategy} ({mode})")
        ax.set(xlabel="Acquired unique pairs", ylabel=label, title=label); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT / name, dpi=160); plt.close(fig)
    write_report(frame, summary)


LOW_BUDGET_EXTENSION_SEEDS = SEEDS + (404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313)


def aggregate_low_budget_extension():
    """Create a standalone 15-seed budget-10 endpoint without pooling other budgets."""
    rows, missing = [], []
    for seed in LOW_BUDGET_EXTENSION_SEEDS:
        for strategy in PRIMARY:
            identifier = job_id({"seed": seed, "budget": 10, "strategy": strategy, "symmetry_mode": "none"})
            path = OUT / "jobs" / identifier / "result.json"
            if not path.exists():
                missing.append(identifier)
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"seed": seed, "strategy": strategy, "budget": 10,
                         "post_test_accuracy": result["post"]["test_accuracy"],
                         "batch_utility": result["batch_utility"], "job_id": identifier})
    if missing:
        raise ValueError("The low-budget extension is incomplete. Run these jobs first:\n" + "\n".join(missing))
    output = RESULT_ROOT / "low_budget_extension_15seed"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_budget10_outcomes.csv", index=False)
    summary = frame.groupby("strategy", as_index=False).agg(
        n=("seed", "count"), post_test_accuracy_mean=("post_test_accuracy", "mean"),
        post_test_accuracy_std=("post_test_accuracy", "std"), batch_utility_mean=("batch_utility", "mean"),
        batch_utility_std=("batch_utility", "std"))
    summary.to_csv(output / "budget10_strategy_summary.csv", index=False)
    atomic_json({"budget": 10, "seeds": LOW_BUDGET_EXTENSION_SEEDS, "strategies": PRIMARY,
                 "source": "budget_curve_study jobs; this is a separate low-budget endpoint", "n": len(frame)},
                output / "study_manifest.json")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strategy, group in frame.groupby("strategy"):
        ax.scatter(group.seed, group.post_test_accuracy, label=strategy, alpha=.8)
        row = summary[summary.strategy == strategy].iloc[0]
        ax.axhline(row.post_test_accuracy_mean, linestyle="--", linewidth=1)
    ax.set(xlabel="Reproducibility seed", ylabel="Post-acquisition fixed ideal-test accuracy",
           title="Budget-10 low-label extension (15 seeds)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "post_test_accuracy_budget10_15seed.png", dpi=160); plt.close(fig)
    return output


def write_report(frame: pd.DataFrame | None = None, summary: pd.DataFrame | None = None):
    if frame is None: frame = pd.read_csv(OUT / "per_seed_results.csv")
    if summary is None: summary = pd.read_csv(OUT / "budget_summary.csv")
    audits = [json.loads(p.read_text(encoding="utf-8"))["audit"] for p in (OUT / "jobs").glob("*/result.json")]
    audit_text = json.dumps(audits[0], indent=2) if audits else "No completed job audits yet."
    text = "# Budget Curve Study Audit\n\n## Protocol\n\nThis audit is generated from immutable per-job manifests. It evaluates single-shot acquisition at fixed epoch 3 and learning rate 1e-4; acquisition budget is the experimental x-axis. Pair partitions are disjoint by unordered pair but may reuse an image across pairs.\n\n## Reproducibility and leakage audit\n\n```json\n" + audit_text + "\n```\n\n## Training\n\nThe model is a SimCLR ResNet-18 encoder with a 512-to-256-to-5 Bradley-Terry reward head. All parameters are trainable. Optimizer and split details are recorded in each `result.json`. Downstream fine-tuning uses affine rotation (+/-5 degrees), translation (+/-5%), scale (0.95-1.05), and brightness/contrast jitter (0.2). The SimCLR pretraining recipe was not available in this repository and is not inferred.\n\n## Results\n\n" + summary.to_markdown(index=False) + "\n\n## Interpretation boundary\n\nThe historical sequential exploration used 70 initial pairs, a 100-pair pool, two-pair batches, and a 30-image test set. It is not pooled with this controlled study, which uses 50 initial pairs, a 120-pair pool, and one acquisition of the stated budget.\n\n## Exploratory features\n\nFeature-space NMF entropy is an exploratory multi-component surrogate, not a physical mixture fraction. Metadata fusion activates only with a populated metadata CSV; the template is in the immutable data directory.\n"
    (OUT / "technical_report.md").write_text(text, encoding="utf-8")


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-manifest"); g.add_argument("--symmetry", action="store_true"); g.add_argument("--smoke", action="store_true"); g.add_argument("--exploratory", action="store_true"); g.add_argument("--budgets", help="Comma-separated acquired-pair budgets, e.g. 10,25."); g.add_argument("--seeds", help="Comma-separated reproducibility seeds."); g.add_argument("--strategies", help="Comma-separated selectors, e.g. random,uncertainty."); g.add_argument("--manifest-name", help="Output manifest name under the pair-disjoint study results."); g.add_argument("--data-root")
    r = sub.add_parser("run-job"); r.add_argument("--manifest", type=Path, default=OUT / "budget_manifest.json"); r.add_argument("--job-id", required=True); r.add_argument("--data-root")
    sub.add_parser("aggregate"); sub.add_parser("aggregate-low-budget-extension"); sub.add_parser("report")
    args = parser.parse_args()
    DATA_ROOT = getattr(args, "data_root", None)
    if args.command == "generate-manifest":
        budgets = parse_csv_values(args.budgets, int, "budgets")
        seeds = parse_csv_values(args.seeds, int, "seeds")
        strategies = parse_csv_values(args.strategies, str, "strategies")
        print(generate_manifest(args.symmetry, args.smoke, args.exploratory, budgets, seeds, strategies, args.manifest_name))
    elif args.command == "run-job": run_job(load_job(args.manifest, args.job_id))
    elif args.command == "aggregate": aggregate()
    elif args.command == "aggregate-low-budget-extension": print(aggregate_low_budget_extension())
    else: write_report()


if __name__ == "__main__": main()

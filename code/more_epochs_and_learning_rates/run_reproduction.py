#!/usr/bin/env python3
"""Strict image-disjoint active-learning reproduction and reporting pipeline.

The implementation deliberately keeps candidate labels in the oracle dataframe and
never passes them to a selector. All generated text and output columns are English.
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
SRC = CODE / "src"
sys.path.insert(0, str(SRC))
from active_learning_pipeline import Config, Experiment, TYPE_ORDER, TYPE_TO_INDEX, canonical_pair, canonical_type
from dataset_protocol import dataset_spec, locate_input


def load_protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_csv(rows: list[dict], path: Path) -> None:
    """Write a completed-cell checkpoint without leaving a partial CSV behind."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def restore_rows(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def progress(label: str, done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed * (total - done) / done if done else float("nan")
    eta_text = "estimating" if not math.isfinite(eta) else f"ETA {eta / 60:.1f} min"
    print(f"[{label}: {done}/{total}, {100 * done / total:.1f}%, elapsed {elapsed / 60:.1f} min, {eta_text}]", flush=True)


def images_for(groups: dict, ids: list[str]) -> set[str]:
    return {p for i in ids for p in groups[i].resolved_img1.tolist() + groups[i].resolved_img2.tolist()}


class StrictExperiment(Experiment):
    """Existing model/augmentation/selector implementation with a strict split."""
    def __init__(self, cfg: Config, protocol: dict, output: Path, split_mode="image_disjoint"):
        super().__init__(cfg); self.protocol, self.output, self.split_mode = protocol, output, split_mode
        self.output.mkdir(parents=True, exist_ok=True); self.validation_rows = pd.DataFrame(); self.holdout_rows = pd.DataFrame()

    def load_and_split(self):
        self.pairwise_csv = locate_input(self.data_root.parent, CODE, self.dataset.pairwise_name)
        self.absolute_csv = locate_input(self.data_root.parent, CODE, self.dataset.absolute_name)
        df = pd.read_csv(self.pairwise_csv)
        required = {"Image1_Path", "Image2_Path", "Reconstruction_Type", "Winner"}
        if not required <= set(df): raise ValueError(f"CSV misses {required - set(df)}")
        df["canonical_type"] = df.Reconstruction_Type.map(canonical_type)
        df = df[df.canonical_type.notna() & df.Winner.astype(str).isin(["1", "2", "tie", "not_apply"])].copy()
        df["resolved_img1"] = [str((self.data_root / p).resolve()) for p in df.Image1_Path]
        df["resolved_img2"] = [str((self.data_root / p).resolve()) for p in df.Image2_Path]
        df = df[df.resolved_img1.map(lambda p: Path(p).is_file()) & df.resolved_img2.map(lambda p: Path(p).is_file())].copy()
        df["pair_id"] = [canonical_pair(a, b) for a, b in zip(df.resolved_img1, df.resolved_img2)]
        df["type_idx"] = df.canonical_type.map(TYPE_TO_INDEX)
        df["confidence_weight"] = df.get("Confidence", pd.Series(index=df.index)).map({"Confident": 1.0, "Somewhat sure": .7}).fillna(1.0)
        all_groups = {k: g.reset_index(drop=True) for k, g in df.groupby("pair_id", sort=True)}
        paths = sorted({*df.resolved_img1, *df.resolved_img2}); rng = random.Random(self.cfg.seed); rng.shuffle(paths)
        cuts = self.protocol["image_split"]; n_train = int(len(paths) * cuts["train"]); n_valid = n_train + int(len(paths) * cuts["validation"])
        image_sets = {"train": set(paths[:n_train]), "validation": set(paths[n_train:n_valid]), "outer_test": set(paths[n_valid:])}
        def pair_split(g):
            pair_paths = set(g.resolved_img1) | set(g.resolved_img2)
            hits = [name for name, members in image_sets.items() if pair_paths <= members]
            return hits[0] if len(hits) == 1 else "cross_partition"
        membership = {pid: pair_split(g) for pid, g in all_groups.items()}
        train_ids = sorted(k for k, v in membership.items() if v == "train")
        self.validation_rows = pd.concat([all_groups[k] for k, v in membership.items() if v == "validation"], ignore_index=True) if "validation" in membership.values() else pd.DataFrame()
        self.holdout_rows = pd.concat([all_groups[k] for k, v in membership.items() if v == "outer_test"], ignore_index=True) if "outer_test" in membership.values() else pd.DataFrame()
        if self.split_mode == "pair_disjoint_sensitivity":
            train_ids = sorted(all_groups); self.validation_rows = pd.DataFrame(); self.holdout_rows = pd.DataFrame()
        self.groups = {k: all_groups[k] for k in train_ids}
        shuffled = train_ids[:]; rng.shuffle(shuffled); initial = shuffled[:min(self.cfg.initial_pairs, len(shuffled))]
        pool = shuffled[len(initial):len(initial) + min(self.cfg.candidate_pairs, max(0, len(shuffled)-len(initial)))]
        self.candidate_metadata = {pid: {"pair_id": pid, "img1": self.groups[pid].iloc[0].resolved_img1, "img2": self.groups[pid].iloc[0].resolved_img2} for pid in pool}
        self._split_ideals(); self._load_bad_references()
        ideal_sets = {"reference": {str(p) for ps in self.references.values() for p in ps}, "ideal_validation": {str(p) for ps in self.utility_images.values() for p in ps}, "ideal_outer_test": {str(p) for ps in self.test_images.values() for p in ps}}
        pair_images = set(paths)
        if any(pair_images & values for values in ideal_sets.values()):
            raise RuntimeError("Pairwise and ideal-image partitions overlap by image ID.")
        self.write_audit(image_sets, membership, initial, pool, ideal_sets)
        if not initial or not pool: raise RuntimeError("Strict split has insufficient eligible training pair groups; see audit.json.")
        return initial, pool

    def write_audit(self, image_sets, membership, initial, pool, ideal_sets):
        base = self.output / "splits" / f"seed-{self.cfg.seed}"; base.mkdir(parents=True, exist_ok=True)
        overlaps = {f"{a}_{b}": sorted(image_sets[a] & image_sets[b]) for a in image_sets for b in image_sets if a < b}
        audit = {"split_mode": self.split_mode, "seed": self.cfg.seed, "image_ids": {k: sorted(v) for k, v in image_sets.items()},
                 "pair_ids": {k: sorted(pid for pid, part in membership.items() if part == k) for k in (*image_sets, "cross_partition")},
                 "row_counts": {"training": int(sum(len(self.groups[x]) for x in self.groups)), "validation": len(self.validation_rows), "outer_test": len(self.holdout_rows)},
                 "initial_pair_ids": initial, "candidate_pair_ids": pool, "image_overlap_checks": overlaps,
                 "exact_pair_overlap_initial_candidate": sorted(set(initial) & set(pool)),
                 "ideal_image_ids": {k: sorted(v) for k, v in ideal_sets.items()},
                 "ideal_image_overlap_checks": {f"{a}_{b}": sorted(ideal_sets[a] & ideal_sets[b]) for a in ideal_sets for b in ideal_sets if a < b},
                 "pairwise_ideal_image_overlap": sorted(set().union(*image_sets.values()) & set().union(*ideal_sets.values())),
                 "candidate_labels_hidden_from_selector": True,
                 "eligible_training_pairs": len(self.groups), "requested_initial_pairs": self.cfg.initial_pairs, "requested_candidate_pairs": self.cfg.candidate_pairs}
        if self.split_mode == "image_disjoint" and (any(overlaps.values()) or audit["exact_pair_overlap_initial_candidate"] or any(audit["ideal_image_overlap_checks"].values()) or audit["pairwise_ideal_image_overlap"]): raise RuntimeError("Leakage audit failed.")
        (base / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def validation_accuracy(self, model):
        return self.evaluate(model, "utility_validation")["test_accuracy"]

    def outer_pairwise_accuracy(self, model):
        return self.pairwise_accuracy(model, self.holdout_rows)


def build_experiment(protocol, seed, output, encoder, lr, epochs, split_mode="image_disjoint", device="auto"):
    cfg = Config(seed=seed, dataset_version=protocol["dataset_version"], epochs=epochs, lr=lr,
                 train_batch_size=protocol["train_batch_size"], weight_decay=protocol["weight_decay"], dropout_p=protocol["dropout_p"],
                 diversity_lambda=protocol["diversity_lambda"], encoder_initialization=encoder, acquisition_mode="single-shot", data_root=None, device=device)
    return StrictExperiment(cfg, protocol, output, split_mode)


def calibrate(protocol, output, smoke=False):
    rows = []
    seeds = protocol["seeds"][:1] if smoke else protocol["seeds"]
    for encoder in protocol["encoder_initializations"]:
        for lr in protocol["learning_rates"]:
            for epochs in protocol["epochs"]:
                for seed in seeds:
                    exp = build_experiment(protocol, seed, output, encoder, lr, epochs); initial, _ = exp.load_and_split()
                    model, train = exp.train(initial)
                    rows.append({"seed": seed, "encoder_initialization": encoder, "learning_rate": lr, "epochs": epochs,
                                 "validation_ideal_accuracy": exp.validation_accuracy(model), "validation_pairwise_accuracy": exp.pairwise_accuracy(model, exp.validation_rows), "training_pairwise_accuracy": train["pairwise_accuracy"]})
    out = output / "calibration"; out.mkdir(parents=True, exist_ok=True); frame = pd.DataFrame(rows); frame.to_csv(out / "grid_results.csv", index=False)
    ranking = frame.groupby(["encoder_initialization", "learning_rate", "epochs"], as_index=False).validation_ideal_accuracy.mean().sort_values(["encoder_initialization", "validation_ideal_accuracy", "learning_rate"], ascending=[True, False, True])
    locks = {encoder: ranking[ranking.encoder_initialization == encoder].iloc[0][["learning_rate", "epochs"]].astype(float).to_dict() for encoder in protocol["encoder_initializations"]}
    (out / "locked_protocol.json").write_text(json.dumps(locks, indent=2), encoding="utf-8")


def audit(protocol, output, smoke=False):
    """Create and fail-closed validate every planned strict split without training."""
    seeds = protocol["seeds"][:1] if smoke else protocol["seeds"]
    rows = []
    for seed in seeds:
        exp = build_experiment(protocol, seed, output, "simclr", 1e-4, 3)
        initial, pool = exp.load_and_split()
        rows.append({"seed": seed, "eligible_training_pairs": len(exp.groups), "initial_pairs": len(initial), "candidate_pairs": len(pool), "validation_rows": len(exp.validation_rows), "outer_test_rows": len(exp.holdout_rows)})
    pd.DataFrame(rows).to_csv(output / "split_capacity.csv", index=False)


def run_experiment(protocol, output, smoke=False, split_mode="image_disjoint", shard_index=0, num_shards=1, device="auto"):
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard-index must be in [0, num-shards).")
    locks = json.loads((output / "calibration" / "locked_protocol.json").read_text(encoding="utf-8"))
    seeds = protocol["seeds"][:1] if smoke else protocol["seeds"]
    budgets = protocol["budgets"][:1] if smoke else protocol["budgets"]
    tasks = [(encoder, seed, budget, update_control, strategy)
             for encoder in sorted(locks) for seed in seeds for budget in budgets
             for update_control in ("fixed_epochs", "fixed_optimizer_updates") for strategy in protocol["strategies"]]
    assigned = [(index, task) for index, task in enumerate(tasks) if index % num_shards == shard_index]
    stem = "raw_results" if split_mode == "image_disjoint" else "raw_pair_disjoint_sensitivity"
    suffix = f".shard-{shard_index}-of-{num_shards}.csv"
    final_path, partial_path = output / (stem + suffix), output / (stem + suffix + ".partial")
    rows = restore_rows(partial_path) if partial_path.exists() else restore_rows(final_path)
    complete = {(r["encoder_initialization"], int(r["seed"]), int(r["requested_budget"]), r["update_control"], r["strategy"]) for r in rows}
    started = time.monotonic(); done = len(rows)
    print(f"Shard {shard_index}/{num_shards} owns {len(assigned)} cells; {done} cells already checkpointed.", flush=True)
    for _, (encoder, seed, budget, update_control, strategy) in assigned:
        key = (encoder, seed, budget, update_control, strategy)
        if key in complete:
            continue
        locked = locks[encoder]
        print(f"Starting cell: encoder={encoder}, seed={seed}, budget={budget}, control={update_control}, strategy={strategy}", flush=True)
        exp = build_experiment(protocol, seed, output, encoder, float(locked["learning_rate"]), int(locked["epochs"]), split_mode, device); initial, pool = exp.load_and_split()
        max_updates = None if update_control == "fixed_epochs" else int(protocol["fixed_optimizer_updates"])
        n = min(budget, len(pool)); baseline, _ = exp.train(initial, max_optimizer_updates=max_updates)
        candidates, cache = exp.candidates_with_clusters(pool, baseline)
        selected, _, _ = exp.select(strategy, candidates, baseline, cache, [], budget=n, labeled_ids=initial)
        chosen = [x["pair_id"] for x in selected]; model, train = exp.train(initial + chosen, max_optimizer_updates=max_updates); ideal = exp.evaluate(model)
        rows.append({"analysis": split_mode, "seed": seed, "encoder_initialization": encoder, "learning_rate": locked["learning_rate"], "epochs": locked["epochs"], "update_control": update_control, "strategy": strategy, "requested_budget": budget, "actual_budget": n, "selected_pair_ids": json.dumps(chosen), "validation_ideal_accuracy": exp.validation_accuracy(model), "validation_pairwise_accuracy": exp.pairwise_accuracy(model, exp.validation_rows), "outer_pairwise_accuracy": exp.outer_pairwise_accuracy(model), "outer_ideal_accuracy": ideal["test_accuracy"], "outer_ideal_correct": ideal["test_correct"], "outer_ideal_total": ideal["test_total"], "outer_ideal_by_class": json.dumps(ideal["by_class"]), "training_pairwise_accuracy": train["pairwise_accuracy"], "optimizer_updates": train["optimizer_updates"]})
        complete.add(key); done += 1; atomic_csv(rows, partial_path); progress(f"shard {shard_index}/{num_shards}", done, len(assigned), started)
    atomic_csv(rows, final_path)


def merge_shards(output: Path, split_mode: str, num_shards: int) -> None:
    stem = "raw_results" if split_mode == "image_disjoint" else "raw_pair_disjoint_sensitivity"
    paths = [output / f"{stem}.shard-{index}-of-{num_shards}.csv" for index in range(num_shards)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot merge: missing completed shard files: " + ", ".join(missing))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["seed", "encoder_initialization", "requested_budget", "update_control", "strategy"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Cannot merge: duplicate experiment cells found across shards.")
    atomic_csv(frame.to_dict("records"), output / f"{stem}.csv")
    print(f"Merged {len(frame)} cells into {stem}.csv.", flush=True)


def bootstrap_ci(values, n=10000, seed=0):
    x = np.asarray(values, float); rng = np.random.default_rng(seed)
    if len(x) == 0: return (np.nan, np.nan)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1); return tuple(np.quantile(means, [.025, .975]))


def paired_permutation(diffs, n, seed):
    x = np.asarray(diffs, float); observed = abs(x.mean()); rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1, 1], size=(n, len(x))) * x).mean(1)); return (1 + int((null >= observed).sum())) / (n + 1)


def holm(rows, alpha):
    ordered = sorted(rows, key=lambda x: x["p_value"]); m = len(ordered)
    for i, row in enumerate(ordered): row["holm_threshold"] = alpha / (m-i); row["holm_reject"] = row["p_value"] <= row["holm_threshold"]


def aggregate(protocol, output):
    frame = pd.read_csv(output / "raw_results.csv"); summary = []
    for keys, group in frame.groupby(["encoder_initialization", "update_control", "strategy", "requested_budget"]):
        lo, hi = bootstrap_ci(group.outer_ideal_accuracy, protocol["bootstrap_replicates"], hash(keys) % 2**32)
        summary.append(dict(zip(["encoder_initialization", "update_control", "strategy", "budget"], keys), n_seeds=len(group), mean_outer_ideal_accuracy=group.outer_ideal_accuracy.mean(), sd_outer_ideal_accuracy=group.outer_ideal_accuracy.std(ddof=1), ci95_low=lo, ci95_high=hi, mean_validation_ideal_accuracy=group.validation_ideal_accuracy.mean(), mean_validation_pairwise_accuracy=group.validation_pairwise_accuracy.mean(), mean_outer_pairwise_accuracy=group.outer_pairwise_accuracy.mean()))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    paired, tests = [], []
    for keys, group in frame.groupby(["encoder_initialization", "update_control", "requested_budget"]):
        pivot = group.pivot(index="seed", columns="strategy", values="outer_ideal_accuracy")
        for strategy in [x for x in pivot.columns if x != "random"]:
            d = (pivot[strategy] - pivot["random"]).dropna(); p = paired_permutation(d, protocol["permutation_replicates"], hash((keys, strategy)) % 2**32)
            paired += [{"seed": seed, "encoder_initialization": keys[0], "update_control": keys[1], "budget": keys[2], "strategy": strategy, "difference_vs_random": value} for seed, value in d.items()]
            tests.append({"encoder_initialization": keys[0], "update_control": keys[1], "budget": keys[2], "strategy": strategy, "n_pairs": len(d), "mean_difference_vs_random": d.mean(), "p_value": p})
    holm(tests, protocol["alpha"]); pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False); pd.DataFrame(tests).to_csv(output / "paired_tests.csv", index=False)
    plot(frame, output); write_report(protocol, output, pd.DataFrame(summary), pd.DataFrame(tests))


def plot(frame, output):
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    for encoder, group in frame.groupby("encoder_initialization"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, part in group.groupby("strategy"):
            for _, seed_part in part.groupby("seed"): ax.plot(seed_part.requested_budget, seed_part.outer_ideal_accuracy, alpha=.18, color=None)
            mean = part.groupby("requested_budget").outer_ideal_accuracy.mean(); ax.plot(mean.index, mean.values, marker="o", label=strategy)
        ax.set(title=f"Strict image-disjoint outer ideal accuracy: {encoder}", xlabel="Acquired pair groups", ylabel="Accuracy"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures / f"{encoder}_curves.png", dpi=180); plt.close(fig)


def write_report(protocol, output, summary, tests):
    significant = tests[tests.holm_reject] if not tests.empty else tests
    text = "# Strict Image-Disjoint Active-Learning Reproduction\n\n## Protocol\n\nThe primary analysis uses mutually exclusive image-ID partitions before pair construction. Only pairs fully within the training partition were eligible for initial labelling or acquisition. Candidate preference labels were hidden from all selectors. Hyperparameters were selected from validation pairwise accuracy only; the outer ideal-image test was reserved for final evaluation.\n\n## Observed results\n\nSee `summary.csv`, `raw_results.csv`, and `paired_differences.csv` for all seed-level values. Curves show individual seed trajectories and mean trajectories; a non-monotonic mean alone is not interpreted as evidence of failure.\n\n## Statistical support\n\n"
    if significant.empty: text += "No non-random selector met the predeclared Holm-Bonferroni-corrected paired permutation criterion in the completed strict image-disjoint results.\n"
    else: text += "Only rows marked `holm_reject=true` in `paired_tests.csv` support a claim of superiority to Random under this protocol.\n"
    text += "\n## Interpretation limits\n\nThe legacy pair-disjoint benchmark is a sensitivity analysis and is not pooled with these results. The legacy Classifier2 83.3% result is not directly comparable unless dataset, split, model, training, and metric equivalence are established. SimCLR may be described as inferior to ImageNet only when the completed matched comparison supports that result consistently.\n"
    (output / "TECHNICAL_REPORT.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["audit", "calibrate", "experiment", "sensitivity", "merge", "aggregate"]); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--config", type=Path, default=HERE / "protocol.json"); parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--num-shards", type=int, default=1); parser.add_argument("--device", default="auto"); parser.add_argument("--sensitivity-merge", action="store_true"); args = parser.parse_args()
    protocol = load_protocol(args.config); output = HERE / "results"; output.mkdir(exist_ok=True)
    if args.action == "audit": audit(protocol, output, args.smoke_test)
    elif args.action == "calibrate": calibrate(protocol, output, args.smoke_test)
    elif args.action == "experiment": run_experiment(protocol, output, args.smoke_test, "image_disjoint", args.shard_index, args.num_shards, args.device)
    elif args.action == "sensitivity": run_experiment(protocol, output, args.smoke_test, "pair_disjoint_sensitivity", args.shard_index, args.num_shards, args.device)
    elif args.action == "merge": merge_shards(output, "pair_disjoint_sensitivity" if args.sensitivity_merge else "image_disjoint", args.num_shards)
    else: aggregate(protocol, output)

if __name__ == "__main__": main()

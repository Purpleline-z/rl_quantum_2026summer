"""Copy task3c result cells for one seed from Drive into the repository.

Run from Colab after mounting Drive. The expected cell ids are recomputed from the
frozen protocol, so only the requested seed's files are copied, byte for byte, and
each copied file is checked against its seed/budget/strategy.

    python colab/copy_task3c_cells_from_drive.py --seed 202 \
        --drive-output /content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe
"""
import argparse, hashlib, json, shutil
from pathlib import Path

STRATEGIES = ("random", "uncertainty", "core_set", "cluster_quota_uncertainty", "uncertainty_diversity",
              "cluster_margin_pairwise", "mc_dropout_probability_variance", "mc_dropout_mutual_information")
BUDGETS = (10, 25, 50, 75, 100)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "code/active_learning_studies/pair_disjoint_not_image_disjoint/results/simclr_three_seed_identity_safe_task3"


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--drive-output", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads((RESULTS / "frozen_task3b_protocol/simclr_three_seed_budget_specific_schedule.json").read_text())["protocol_by_budget"]
    selector = json.loads((RESULTS / "frozen_selector_parameters/frozen_global_selector_parameters.json").read_text())["global_selector_parameters"]
    source, destination = args.drive_output / "task3c_final_strategy_cells", RESULTS / "task3c_final_strategy_cells"
    copied = existing = 0
    missing = []
    for budget in BUDGETS:
        for strategy in STRATEGIES:
            task = {"phase": "task3c_final", "seed": args.seed, "budget": budget, "strategy": strategy,
                    **schedule[str(budget)], "selector_parameters": selector}
            name = f"{stable_id(task)}.json"
            if (destination / name).exists():
                existing += 1
            elif not (source / name).exists():
                missing.append((budget, strategy, name))
            else:
                cell = json.loads((source / name).read_text())
                assert (cell["seed"], cell["budget"], cell["strategy"]) == (args.seed, budget, strategy), name
                shutil.copyfile(source / name, destination / name)
                copied += 1
    print(f"seed {args.seed}: copied {copied}, already present {existing}, missing on Drive {len(missing)}")
    for item in missing:
        print("  missing:", item)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

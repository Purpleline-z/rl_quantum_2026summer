#!/usr/bin/env python3
"""Emit the markdown tables used in the report from the aggregated endpoint CSVs (no hand-copied numbers)."""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "results" / "pair_endpoint_study"
NAMES = {"random": "Random", "uncertainty": "Uncertainty", "core_set": "Core-set", "cluster_quota_uncertainty": "Cluster-quota uncertainty", "uncertainty_diversity": "Uncertainty + diversity",
         "cluster_margin_pairwise": "Cluster-Margin", "mc_dropout_probability_variance": "MC-dropout variance", "mc_dropout_mutual_information": "MC-dropout mutual info",
         "core_set_relation": "Core-set, relation-aware pairs", "typiclust_pairs": "TypiClust (pairs)", "badge_pairs": "BADGE (pairs)", "fisher_dopt": "Fisher D-optimal (Active Reward Modeling)",
         "image_coverage_uncertainty": "Image-coverage uncertainty", "graph_facility_location": "Graph facility location (uncertainty-weighted)", "uncertainty_all_heads": "Uncertainty, all heads",
         "delta_gap": "Largest predicted gap", "delta_ucb": "Gap + posterior std (DeltaUCB-style)", "bald_decisive": "BALD x P(decisive)", "dpp_pairs": "DPP (quality x diversity)",
         "probcover_pairs": "ProbCover (pairs)", "maxherding_pairs": "MaxHerding (pairs)", "laplace_bald": "Laplace BALD", "dropquery_pairs": "DropQuery (pairs)", "fass_pairs": "FASS (pairs)", "graphcut_pairs": "Graph cut (pairs)",
         "ensemble_bald": "Deep-ensemble BALD (8 heads)", "ensemble_bald_decisive": "Deep-ensemble BALD x P(decisive)",
         "uncertainty_lf": "Uncertainty, label-free", "cluster_quota_uncertainty_lf": "Cluster-quota uncertainty, label-free", "uncertainty_diversity_lf": "Uncertainty + diversity, label-free",
         "cluster_margin_pairwise_lf": "Cluster-Margin, label-free", "mc_dropout_probability_variance_lf": "MC-dropout variance, label-free", "mc_dropout_mutual_information_lf": "MC-dropout mutual info, label-free"}


def p(value) -> str:
    return "–" if pd.isna(value) else ("<0.001" if value < .001 else f"{value:.3f}")


def main() -> None:
    mode = sys.argv[1]; seeds = sys.argv[2]; tag = f"{mode}_{seeds}"
    tables = {}
    for metric, label in (("test_decisive_log_loss", "log-loss"), ("test_calibrated_log_loss", "cal. log-loss"), ("test_decisive_auc", "AUC")):
        path = OUT / f"{tag}_{metric}_vs_random.csv"
        if path.exists(): tables[label] = pd.read_csv(path, index_col=0)
    if not tables: raise SystemExit("run aggregate_pair_endpoint.py first")
    base = tables["log-loss"]; ranked = base.index
    lines = ["| Strategy | log-loss gain | Holm p | cal. log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for family in ranked:
        row = [NAMES.get(family, family)]
        for label in ("log-loss", "cal. log-loss", "AUC"):
            t = tables[label]
            row += [f"{t.loc[family, 'mean_gain']:+.3f}" if family in t.index else "–", p(t.loc[family, "holm_p"]) if family in t.index else "–"]
        row.append(f"{base.loc[family, 'frac_better']:.0%} ({int(base.loc[family, 'n'])})"); lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

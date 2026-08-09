#!/usr/bin/env python3
"""Summarize non-additive batch effects from fixed-protocol jobs."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from project_paths import RESULT_ROOT

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage", default="final_comparison"); a = p.parse_args()
    source = RESULT_ROOT / "fixed_protocol_3seed" / a.stage / "jobs"; out = RESULT_ROOT / "fixed_protocol_3seed" / "diagnostics"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in source.glob("*/result.json"):
        result = json.loads(file.read_text()); values = result.get("individual_utilities")
        if values:
            utilities = np.asarray([x["delta_accuracy"] if isinstance(x, dict) else x for x in values], dtype=float)
            result["mean_individual_utility"] = float(utilities.mean()); result["sum_individual_utility"] = float(utilities.sum()); result["interaction_gap"] = float(result["batch_utility"] - utilities.sum()); result["individual_utility_available"] = True
        else:
            result["mean_individual_utility"] = np.nan; result["sum_individual_utility"] = np.nan; result["interaction_gap"] = np.nan; result["individual_utility_available"] = False
        result["job_path"] = str(file.parent); rows.append(result)
    if not rows: raise SystemExit("No completed result.json files found.")
    flat = pd.json_normalize(rows); flat.to_csv(out / "batch_interaction_diagnostics.csv", index=False)
    metadata = {"warning": "The fixed ideal test set is small; accuracy changes are coarse and noisy. Missing individual utilities are reported as unavailable, never estimated.", "source_stage": a.stage}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    available = flat.dropna(subset=["interaction_gap"])
    for metric in ("image_reuse_rate", "cluster_entropy", "mean_pair_cosine_similarity"):
        if not available.empty and metric in available:
            fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(available[metric], available["interaction_gap"])
            ax.set(xlabel=metric, ylabel="interaction_gap = batch utility - sum individual utilities", title="Batch interaction diagnostic"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / f"interaction_gap_vs_{metric}.png", dpi=160); plt.close(fig)
    print(out / "batch_interaction_diagnostics.csv")


if __name__ == "__main__": main()

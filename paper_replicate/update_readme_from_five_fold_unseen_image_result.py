"""Insert only the current five-fold unseen-image study interpretation into the README."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


START = "<!-- FIVE_FOLD_UNSEEN_IMAGE_RESULTS_START -->"
END = "<!-- FIVE_FOLD_UNSEEN_IMAGE_RESULTS_END -->"


def text_for(result):
    if result["status"] != "completed":
        return "No completed five-fold classifier result is available for this run."
    rows = []
    for variant, summary in result["architecture_test_summary"].items():
        accuracy = summary["decisive_pairwise_accuracy"]
        macro_f1 = summary["all_label_macro_f1"]
        rows.append(f"| {variant} | {accuracy['mean']:.3f} [{accuracy['bootstrap_95_percent_interval'][0]:.3f}, {accuracy['bootstrap_95_percent_interval'][1]:.3f}] | {macro_f1['mean']:.3f} [{macro_f1['bootstrap_95_percent_interval'][0]:.3f}, {macro_f1['bootstrap_95_percent_interval'][1]:.3f}] |")
    return ("This table is generated only from the completed five-fold image-identity study. Each test pair contains at least one image excluded from pairwise reward-model fine-tuning, validation, calibration, and architecture selection. The frozen laboratory RHEED-SimCLR encoder was pretrained without pairwise preference labels; it is an in-domain self-supervised representation rather than a fully inductive, unseen-pixel pretraining benchmark.\n\n"
            "| Architecture | Mean decisive accuracy [95% bootstrap interval] | Mean all-label macro-F1 [95% bootstrap interval] |\n"
            "|---|---:|---:|\n" + "\n".join(rows) + "\n\n"
            f"The deployment architecture is **{result['selected_deployment_architecture']}**, selected by mean validation decisive-pair accuracy across all outer folds. " + result["next_research_step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    readme = args.readme.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        raise ValueError("README result markers are missing.")
    before, rest = readme.split(START, 1)
    _, after = rest.split(END, 1)
    args.readme.write_text(before + START + "\n\n" + text_for(result) + "\n\n" + END + after, encoding="utf-8")


if __name__ == "__main__": main()

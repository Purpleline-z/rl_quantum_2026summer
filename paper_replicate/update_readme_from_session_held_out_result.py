"""Insert only the current session-held-out study interpretation into the README."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


START = "<!-- SESSION_HELD_OUT_RESULTS_START -->"
END = "<!-- SESSION_HELD_OUT_RESULTS_END -->"


def text_for(result):
    if result["status"] != "completed":
        return ("No classifier result is reported for this run. The data-readiness audit stopped before training because one or more sealed sessions had fewer than 50 decisive pairs. "
                "The published audit identifies the exact label gap; collecting those comparisons is the next research action.")
    rows = []
    for fold in result["outer_fold_results"]:
        rows.append(f"| {fold['held_out_test_session']} | {fold['selected_architecture']} | {fold['mean_sealed_decisive_accuracy']:.3f} | {fold['mean_sealed_macro_f1']:.3f} |")
    return ("The table below is generated only from the latest completed three-fold session-held-out run. "
            "Each reported test session was excluded from training, validation, architecture selection, and calibration.\n\n"
            "| Held-out session | Validation-selected architecture | Mean decisive accuracy across five seeds | Mean all-label macro-F1 |\n"
            "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
            f"Across outer folds, mean decisive accuracy is **{result['mean_outer_fold_decisive_accuracy']:.3f}** and mean macro-F1 is **{result['mean_outer_fold_macro_f1']:.3f}**. "
            + result["next_research_step"])


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

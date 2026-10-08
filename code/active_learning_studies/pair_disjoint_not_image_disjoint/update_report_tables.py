#!/usr/bin/env python3
"""Regenerate the strategy tables embedded in report section 5.12 from the aggregated CSVs (keeps names and numbers in sync)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "ACADEMIC_REPORT_DRAFT.md"
sys.path.insert(0, str(HERE))
from make_endpoint_report_tables import NAMES  # noqa: E402


def gain_table(mode: str) -> str:
    return subprocess.run([sys.executable, str(HERE / "make_endpoint_report_tables.py"), mode, "all"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()


def budget_table(mode: str, csv: str = "by_budget_gain_vs_random.csv") -> str:
    d = pd.read_csv(HERE / "results" / "pair_endpoint_study" / csv); lines = ["| Budget | Metric | Three largest gains over Random (Holm p) | Significant after Holm correction within this cell |", "|---:|---|---|---|"]
    for budget in (10, 20, 40, 60):
        for metric in ("log-loss", "AUC"):
            h = d[(d["mode"] == mode) & (d.metric == metric) & (d.budget == budget)].sort_values("mean_gain", ascending=False)
            top = "; ".join(f"{NAMES.get(r.family, r.family)} {r.mean_gain:+.3f} ({r.holm_p:.2f})" for r in h.head(3).itertuples())
            sig = "; ".join(f"{NAMES.get(r.family, r.family)} {r.mean_gain:+.3f}" for r in h[h.holm_p < .05].itertuples()) or "none"
            lines.append(f"| {budget} | {metric} | {top} | {sig} |")
    return "\n".join(lines)


def swap(text: str, marker: str, table: str) -> str:
    i = text.index(marker); i = text.index("| ", i); j = text.index("\n\n", i); return text[:i] + table + text[j:]


def main() -> None:
    t = REPORT.read_text()
    t = swap(t, "**Results: single-shot, 35 seeds.**", gain_table("single")); t = swap(t, "**Results: sequential, 35 seeds.**", gain_table("sequential"))
    t = swap(t, "*Single-shot* (one batch of the budget size chosen from the 10-group model):", budget_table("single")); t = swap(t, "*Sequential* (rounds of 10 groups):", budget_table("sequential"))
    REPORT.write_text(t)


if __name__ == "__main__":
    main()

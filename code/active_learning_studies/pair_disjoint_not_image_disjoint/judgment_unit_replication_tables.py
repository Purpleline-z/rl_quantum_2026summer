#!/usr/bin/env python3
"""Markdown tables for JUDGMENT_UNIT_REPLICATION_RESULTS.md, generated from the analysis CSVs written by judgment_unit_replication_analysis.py (no hand-copied numbers).
usage: judgment_unit_replication_tables.py confirmatory | h6 | pooled | full <metric: log-loss|AUC|accuracy> | verdicts | pooled_verdicts"""
import sys
import pandas as pd
import judgment_unit_replication_analysis as ra

A = ra.OUT; N = ra.NAMES
fp = lambda v: "<0.001" if v < .001 else f"{v:.3f}"
cell = lambda s, c: f"{s} {'single' if c == 'single' else 'seq.'}"


def tests(csv: str, main_csv: str | None) -> str:
    t = pd.read_csv(A / csv); m = pd.read_csv(A / main_csv) if main_csv else None
    head = "| Hyp. | Variant | Cell | Metric | Direction | Estimate | Seeds in predicted direction | one-sided p | Holm p | Replicated? | " + ("Main-run estimate |" if m is not None else "")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for i, r in t.iterrows():
        share = r.share_seeds_gain_gt0 if r.direction == "greater" else r.share_seeds_gain_lt0
        row = [r.hypothesis, N[r.variant], cell(r.split, r.condition), r.metric, "gain > 0" if r.direction == "greater" else "gain < 0", f"{r.mean_gain:+.3f}", f"{share:.0%}", fp(r.p_one_sided), fp(r.holm_p), "yes" if r.test_replicated else "no"]
        if m is not None: row.append(f"{m.loc[i, 'mean_gain']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def verdicts(csv: str) -> str:
    v = pd.read_csv(A / csv); lines = ["| Hyp. | Claim | Tests | Replicated tests (Holm p < 0.05, right sign) | Tests with right sign | Verdict |", "|---|---|---:|---:|---:|---|"]
    for _, r in v.iterrows(): lines.append(f"| {r.hypothesis} | {N.get(r.claim, r.claim)} | {r.tests} | {r.replicated_tests} | {r.right_sign_tests} | {r.verdict} |")
    return "\n".join(lines)


def full(metric: str) -> str:
    cells = ra.CELLS; head = "| Variant | " + " | ".join(f"{cell(s, c)}: new / main" for s, c in cells) + " |"; lines = [head, "|---|" + "---:|" * len(cells)]
    tabs = {k: pd.read_csv(A / f"full_table_{k[0]}_{k[1]}_{metric}.csv", index_col=0) for k in cells}
    for v in ra.VARIANTS + [ra.EXTRA]:
        row = []
        for k in cells:
            t = tabs[k]; star = "*" if v != ra.EXTRA and t.loc[v, "holm_p_over_13_new"] < .05 else ""
            row.append(f"{t.loc[v, 'mean_gain_new']:+.3f}{star} ({t.loc[v, 'share_better_new']:.0%}) / {t.loc[v, 'mean_gain_main']:+.3f}")
        lines.append(f"| {N[v]} | " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    a = sys.argv[1]
    print({"confirmatory": lambda: tests("confirmatory_tests.csv", "main_seeds_same_tests.csv"), "h6": lambda: tests("h6_null_check_tests.csv", None),
           "pooled": lambda: tests("exploratory_pooled_tests.csv", None), "pooled_h6": lambda: tests("exploratory_pooled_h6_tests.csv", None), "verdicts": lambda: verdicts("confirmatory_verdicts.csv"),
           "pooled_verdicts": lambda: verdicts("exploratory_pooled_verdicts.csv"), "full": lambda: full(sys.argv[2])}[a]())

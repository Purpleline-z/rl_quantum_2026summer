"""Pre-registered confirmation analysis of the recipe study (rules frozen in PREREGISTRATION_RC1.md before any confirmation seed was run).

  python3 rc_confirm.py DIR [DIR ...] --ref lap3_k10_st4 --cands a,b,c,d [--control aw8_all_st2lr0.5] [--out FILE]

Families (Holm within each family; per seed the gain is the mean over budgets and draws of the paired difference on identical label sets; two-sided Wilcoxon over seeds; paired bootstrap interval):
  main     candidates x splits {A, B} x {auc, acc}                  (16 tests for 4 candidates)   -> verdict per candidate
  htr      candidates x splits x {htr_acc, htr_ll}                  (separate family)               -> verdict per candidate
  far      candidates x splits x {far_auc, far_acc}                 (anchor-far secondary family)   -> flag only
Verdict (main): SUCCESS = all four gains positive AND >= 3 of the 4 Holm p < 0.05.  The same rule is applied to the htr family (HTR improvement) and to the far family (anchor-far gain).
A candidate that succeeds on `main` but not on `far` is reported as "gain only (or mostly) in the anchor-near stratum".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rc_tables as T

FAMILIES = {"main": ("auc", "acc"), "htr": ("htr_acc", "htr_ll"), "far": ("far_auc", "far_acc")}


def family_table(df: pd.DataFrame, ref: str, cands: list[str], metrics) -> pd.DataFrame:
    rows = []
    for c in cands:
        for split in ("A", "B"):
            for m in metrics:
                s = T.summarize(T.per_seed_gain(df, c, ref, split, m)); rows.append({"candidate": c, "split": split, "metric": m, **s})
    out = pd.DataFrame(rows); out["holm"] = T.holm(np.nan_to_num(out.p.to_numpy(), nan=1.0)); return out


def verdicts(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in tab.groupby("candidate", sort=False):
        pos = bool((g.gain > 0).all()); sig = int((g.holm < 0.05).sum()); rows.append({"candidate": c, "all_gains_positive": pos, "holm<0.05 (of 4)": sig, "SUCCESS": bool(pos and sig >= 3)})
    return pd.DataFrame(rows)


def run(dirs, ref, cands, control=None) -> str:
    df = T.load(dirs); out = []
    out.append(f"Seeds per learner/split (cells = budget x draw x seed):\n\n{T.md(df.groupby(['learner', 'split']).agg(seeds=('seed', 'nunique'), cells=('auc', 'size')).reset_index(), fmt='.0f')}\n")
    res = {}
    for fam, metrics in FAMILIES.items():
        tab = family_table(df, ref, cands, metrics); v = verdicts(tab); res[fam] = v
        out.append(f"### Family `{fam}` (Holm over {len(tab)} tests; gain = candidate - reference, positive = better)\n\n{T.md(tab)}\n\nVerdicts:\n\n{T.md(v)}\n")
    merged = res["main"].merge(res["far"][["candidate", "all_gains_positive", "holm<0.05 (of 4)", "SUCCESS"]], on="candidate", suffixes=("", "_far")).merge(
        res["htr"][["candidate", "all_gains_positive", "holm<0.05 (of 4)", "SUCCESS"]], on="candidate", suffixes=("", "_htr"))
    merged["reading"] = [("pre-registered success on the standard endpoint and on the anchor-far stratum" if r.SUCCESS and r.SUCCESS_far else
                          "success on the standard endpoint, but the gain is only (or mostly) in the anchor-near stratum" if r.SUCCESS else "no confirmed gain on the standard endpoint") for r in merged.itertuples()]
    out.append(f"### Summary\n\n{T.md(merged)}\n")
    out.append("### Per-type effects (candidate - reference; accuracy and log-loss gains per type, uncorrected, descriptive)\n\n" + T.md(T.types_table(df, ref, cands)) + "\n")
    out.append("### Anchor strata (AUC and accuracy gain by distance of the test judgment's nearer image to the nearest typed anchor; descriptive)\n\n" + T.md(T.strata_table(T.load(dirs, keep_d=True), ref, cands)) + "\n")
    if control:
        rows = []
        for split in ("A", "B"):
            for m in ("auc", "acc", "far_auc", "far_acc", "htr_acc", "htr_ll"): rows.append({"comparison": f"{ref} - {control}", "split": split, "metric": m, **T.summarize(T.per_seed_gain(df, ref, control, split, m))})
        out.append("### Context: the incumbent against the graph-free control on these seeds (not part of the families; `far_*` = anchor-far stratum)\n\n" + T.md(pd.DataFrame(rows)) + "\n")
        out.append("### Context: where the incumbent's gain over the graph-free control lives (anchor strata, descriptive)\n\n" + T.md(T.strata_table(T.load(dirs, keep_d=True), control, [ref])) + "\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("dirs", nargs="+"); ap.add_argument("--ref", default="lap3_k10_st4"); ap.add_argument("--cands", required=True)
    ap.add_argument("--control"); ap.add_argument("--out", type=Path); a = ap.parse_args()
    dirs = [str((T.NEW / d) if not Path(d).exists() else d) for d in a.dirs]; text = run(dirs, a.ref, a.cands.split(","), a.control)
    if a.out: a.out.write_text(text)
    else: print(text)


if __name__ == "__main__": main()

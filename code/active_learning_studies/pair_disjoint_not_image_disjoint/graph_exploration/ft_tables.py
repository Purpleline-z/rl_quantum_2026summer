"""Tables of the CPU fine-tuning study (generated from the result JSONs of ``new_directions/ft_run.py``; nothing is typed by hand).

  python3 ft_tables.py dev DIR [DIR ...] --ref lap3_k10_st4 [--out FILE]     compact development view: per learner and split the mean paired gain over the reference
                                                                              (AUC, accuracy, anchor-far AUC / accuracy, log-loss [positive = better]), seeds, mean fit seconds
Gain = learner minus reference on identical label sets, mean over budgets and draws, then over seeds; far = both test images > 0.03 (1 - cos) from every typed ideal-image anchor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rc_tables as T


def dev_view(df: pd.DataFrame, ref: str, learners=None) -> pd.DataFrame:
    learners = [l for l in df.learner.unique() if l != ref] if learners is None else learners; rows = []
    for l in learners:
        for split in sorted(df.split.unique()):
            row = {"learner": l, "split": split}
            for m in ("auc", "acc", "far_auc", "far_acc", "ll"):
                s = T.summarize(T.per_seed_gain(df, l, ref, split, m)); row[m] = s["gain"]; row["n_seeds"] = s["n"]
                if m in ("auc", "acc"): row[f"{m}_p"] = s["p"]
            sec = df[(df.learner == l) & (df.split == split)].get("seconds"); row["sec_per_fit"] = float(sec.mean()) if sec is not None and len(sec) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("mode", choices=["dev"]); ap.add_argument("dirs", nargs="+"); ap.add_argument("--ref", default="lap3_k10_st4"); ap.add_argument("--learners"); ap.add_argument("--out", type=Path)
    a = ap.parse_args(); df = T.load([str((T.NEW / d) if not Path(d).exists() else d) for d in a.dirs]); text = T.md(dev_view(df, a.ref, a.learners.split(",") if a.learners else None))
    if a.out: a.out.write_text(text + "\n")
    else: print(text)


if __name__ == "__main__": main()

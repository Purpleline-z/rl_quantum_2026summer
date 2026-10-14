"""HTR failure analysis, Stage 2 (post-hoc family): add the anchor-similarity score of the HTR column to the stored predictions of a learner.

  python3 hf_blend.py --a1 results/hf/a1 --testmap results/hf/testmap_dev.csv --base lap3_k10_st4 --gammas 2,5,10 --out results/hf/blend_dev

For a test row of type HTR (k = 4) with images (a, b): d' = d + gamma * (dHTR(b) - dHTR(a)) where dHTR(x) = 1 - max cosine (centred SimCLR features) between x and the HTR ideal images that the learner already uses as
anchors (references + utility + outer-test ideals of the seed; ``hf_table.py`` precomputes it for the 29 HTR ideals).  Rows of every other type are unchanged, so the other types cannot change by construction.
The output has the rt_a1 format (learner name ``hfb_g<gamma>``), so ``hf_gain.py`` compares it with the base learner on identical label sets.  Because lap3_k10_st4 predictions are deterministic given the label set
(verified: the stored seed-8800 predictions are reproduced to 0.0), blending stored predictions equals refitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import nd_core as core

HF = core.HERE / "results" / "hf"


def blend_dir(a1: Path, testmap: Path, base: str, gammas: list[float], out: Path, score_col: str = "d_HTR", tag: str = "hfb_g") -> None:
    out.mkdir(parents=True, exist_ok=True); tm = pd.read_csv(testmap); j = pd.read_csv(HF / "judgments.csv").set_index("jid")
    for f in sorted(a1.glob("seed*_*.csv.gz")):
        d = pd.read_csv(f); d = d[d.learner == base].merge(tm, on=["seed", "split", "j"], how="left", validate="m:1"); assert d.jid.notna().all()
        gap = (j.loc[d.jid.astype(int), f"{score_col}2"].to_numpy() - j.loc[d.jid.astype(int), f"{score_col}1"].to_numpy()); htr = (d.k == 4).to_numpy(); frames = [d.drop(columns=["jid"])]
        for g in gammas:
            e = d.drop(columns=["jid"]).copy(); e["d"] = e.d + g * gap * htr; e["learner"] = f"{tag}{g:g}"; frames.append(e)
        pd.concat(frames).to_csv(out / f.name, index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a1", type=Path, required=True); ap.add_argument("--testmap", type=Path, required=True); ap.add_argument("--base", default="lap3_k10_st4"); ap.add_argument("--gammas", default="2,5,10"); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); blend_dir(a.a1, a.testmap, a.base, [float(x) for x in a.gammas.split(",")], a.out); print("written", a.out)


if __name__ == "__main__": main()

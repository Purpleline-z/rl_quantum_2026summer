"""HTR failure analysis: contact sheets of judgments (script-generated PNGs).

  python3 hf_sheets.py --per-judgment results/hf/errors_dev_per_judgment.csv --out results/hf/sheets --worst 8 --best 4

Each row = one HTR judgment: image 1 | image 2, each followed by the nearest ideal image (any of the four active classes; title = class, distance); the row title gives the label (which image won the HTR comparison),
the learner's mean margin / error rate over the stored predictions, the model-free rule, labeller, confidence and temperatures.  Winner frames get a green frame, loser frames a red frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import nd_core as core
import rt_core as rt

HF = core.HERE / "results" / "hf"; ROOT = core.single_study.DATA.resolve() / "original data"


def load_img(rel: str):
    with Image.open(ROOT / rel) as im: return np.asarray(im.convert("L"), dtype=float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-judgment", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--worst", type=int, default=8); ap.add_argument("--best", type=int, default=4)
    ap.add_argument("--jids", default=""); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    pj = pd.read_csv(a.per_judgment).set_index("jid"); j = pd.read_csv(HF / "judgments.csv").set_index("jid"); img = pd.read_csv(HF / "images.csv").set_index("img")
    cache = core.feature_cache(); sim = rt.Sim(cache)
    ideals = sorted(p for p in cache if "STO_ideal_" in str(p) and "Twinned" not in str(p)); iidx = [sim.index[p] for p in ideals]
    def nearest_ideal(rel):
        full = str((ROOT / rel).resolve()); s = (sim.xn[sim.index[full]] @ sim.xn[iidx].T).numpy(); k = int(np.argmax(s)); return str(Path(ideals[k]).relative_to(ROOT)), 1 - float(s[k])
    if a.jids: chosen = [int(x) for x in a.jids.split(",")]
    else:
        order = pj.sort_values(["err_rate", "mean_ll"], ascending=False); chosen = list(order.index[:a.worst]) + list(order.index[::-1][:a.best])
    per = 4; sheets = [chosen[i:i + per] for i in range(0, len(chosen), per)]
    for n, group in enumerate(sheets):
        fig, axes = plt.subplots(len(group), 4, figsize=(13, 3.4 * len(group)), squeeze=False)
        for r, jid in enumerate(group):
            row = j.loc[jid]; info = pj.loc[jid] if jid in pj.index else None; win = 1 if row.sign > 0 else 2
            items = [(row.img1, 1), (row.img2, 2)]
            for c, (rel, side) in enumerate(items):
                near, dist = nearest_ideal(rel); ax = axes[r, 2 * c]; ax.imshow(load_img(rel), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
                color = "#2a9d3f" if side == win else "#c0392b"
                for sp in ax.spines.values(): sp.set_edgecolor(color); sp.set_linewidth(4)
                ax.set_title(f"img{side} {'WINNER' if side == win else 'loser'}  {Path(rel).name}\nT={img.loc[rel, 'temp']:.0f}C  d_HTR={img.loc[rel, 'd_HTR']:.3f}  d_RT13={img.loc[rel, 'd_RT13']:.3f}", fontsize=8, color=color)
                ax2 = axes[r, 2 * c + 1]; ax2.imshow(load_img(near), cmap="gray"); ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title(f"nearest ideal: {Path(near).parent.name.replace('STO_ideal_', '')}/{Path(near).name}\n1-cos={dist:.4f}", fontsize=8)
            txt = f"jid {jid}  {row.type} judgment, winner = image {win}; labeller {row.labeller}, conf {row.confidence}"
            if info is not None: txt += f"; learner error rate {info.err_rate:.2f} (mean margin {info.mean_margin:+.2f}, {int(info.n_rows)} rows); rule 'closer to ideal HTR wins' {'right' if info.rule_ok else 'WRONG'}"
            axes[r, 0].annotate(txt, xy=(0, 1.38), xycoords="axes fraction", fontsize=9, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(a.out / f"sheet_{n:02d}.png", dpi=75); plt.close(fig)
    print(len(sheets), "sheets ->", a.out)


if __name__ == "__main__": main()

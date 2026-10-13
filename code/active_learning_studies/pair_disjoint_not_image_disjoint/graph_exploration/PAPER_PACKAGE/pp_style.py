"""Shared figure style (matplotlib).  Colours follow the entity (learner), never the rank; the first six slots of the validated categorical palette (checked with the dataviz validator: lightness band, chroma
floor, adjacent CVD separation and normal-vision floor PASS; contrast of aqua / yellow / magenta on the light surface is < 3:1, therefore every line is direct-labelled and every figure has a legend and a
table behind it in tables/*.csv).  Thin marks, recessive grid, text in ink colours only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pp_lib as L

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"baseline": "#2a78d6", "aw8": "#eb6834", "aw8_all": "#1baf7a", "aw8_all_st2lr0.5": "#eda100", "lap3_k10": "#e87ba4", "lap3_k10_st4": "#008300", "lap3_k10_st4_shuf": "#8a8985"}
LABEL = {"baseline": "default head", "aw8": "aw8", "aw8_all": "aw8_all", "aw8_all_st2lr0.5": "aw8_all_st2lr0.5 (tuned graph-free)", "lap3_k10": "lap3_k10", "lap3_k10_st4": "lap3_k10_st4", "lap3_k10_st4_shuf": "lap3_k10_st4, shuffled graph"}
SPLIT_COL = {"A": "#2a78d6", "B": "#eb6834"}
SPLIT_MARK = {"A": "o", "B": "s"}


def setup():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "xtick.minor.visible": False, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5, "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 100})


def logx(ax, lo=8, hi=60):
    """budget axis: log scale with only the four budgets as ticks (no minor-tick labels)"""
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log"); ax.set_xticks([10, 20, 40, 60]); ax.set_xticklabels(["10", "20", "40", "60"]); ax.xaxis.set_minor_formatter(NullFormatter()); ax.minorticks_off(); ax.set_xlim(lo, hi)


def spread(ax, items, gap=0.045):
    """direct labels at the right end of lines without collisions: items = [(x, y, text, colour)], labels are pushed apart vertically (axes fraction `gap`) and connected by their own colour."""
    y0, y1 = ax.get_ylim(); items = sorted(items, key=lambda t: t[1]); pos = []
    for x, y, text, col in items:
        yy = y if not pos else max(y, pos[-1] + gap * (y1 - y0)); pos.append(yy)
    shift = (pos[-1] - items[-1][1]) / 2 if pos and pos[-1] > items[-1][1] else 0
    for (x, y, text, col), yy in zip(items, pos):
        ax.annotate(text, (x, y), xytext=(x * 1.08, yy - shift), textcoords="data", fontsize=6.8, color=INK2, va="center", annotation_clip=False, arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=1))


def save(fig, name, note=None, note_y=-0.115):
    """png + pdf into figures/; `note` (data source) is written under the figure."""
    L.FIGS.mkdir(exist_ok=True)
    if note: fig.text(0.5, note_y, note, fontsize=6.8, color=INK2, ha="center", va="top")
    for ext in ("png", "pdf"): fig.savefig(L.FIGS / f"{name}.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig); print("figure", name)

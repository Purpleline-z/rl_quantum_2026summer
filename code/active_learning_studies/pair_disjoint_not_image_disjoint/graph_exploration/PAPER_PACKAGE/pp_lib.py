"""Shared helpers of the paper package: paths, result loading, paired gains, bootstrap/Wilcoxon/Holm (same conventions as new_directions/nd_confirm.py), and the number store that every table script
writes to (tables/numbers.json) and that check_prose.py verifies the prose against.  No torch needed; only numpy, pandas, scipy, scikit-learn, matplotlib."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PKG = Path(__file__).resolve().parent
GE = PKG.parent                                   # .../graph_exploration
STUDY = GE.parent                                 # .../pair_disjoint_not_image_disjoint
STUDIES = STUDY.parent                            # .../code/active_learning_studies
ND = STUDIES / "new_directions"
RES = ND / "results"
TABLES = PKG / "tables"; FIGS = PKG / "figures"; BUILD = PKG / "build"; PROSE = PKG / "prose"
for p in (GE, ND, STUDIES.parent / "active_learning_program"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
SIGN = {"auc": 1, "acc": 1, "ll": -1}              # log-loss gain = reference - cell, positive = better (as in nd_confirm.py)
N_BOOT = 4000


def sign_of(metric: str) -> int:
    return -1 if metric.endswith("ll") else 1


@lru_cache(maxsize=None)
def load_dir(rel: str, partial_ok: bool = False) -> pd.DataFrame:
    """All preference records (list-of-dict JSON files seed*_*.json) of results/<rel>; files with fewer records than the fullest file of the directory (cells still being written) are skipped."""
    files = {f: json.loads(f.read_text()) for f in sorted((RES / rel).glob("seed*_*.json"))}
    files = {f: v for f, v in files.items() if isinstance(v, list)}
    if not files: return pd.DataFrame()
    full = max(len(v) for v in files.values())
    rows = [r for v in files.values() if partial_ok or len(v) == full for r in v if isinstance(r, dict)]
    df = pd.DataFrame(rows)
    if "selector" not in df: df["selector"] = "random"
    return df


def per_seed(frame: pd.DataFrame, split: str, metric: str, a, b, budgets=None, by_budget=False):
    """Paired per-seed gain of cell a over cell b (cells = (selector, learner)); mean over draws and budgets (or per budget); sign +1 for auc/acc, -1 for log-loss.  Same as nd_confirm.gains."""
    g = frame[frame.split == split]
    if budgets is not None: g = g[g.budget.isin(budgets)]
    mean = lambda cell: g[(g.selector == cell[0]) & (g.learner == cell[1])].groupby(["seed", "budget"])[metric].mean()
    d = (sign_of(metric) * (mean(a) - mean(b))).dropna()
    return d if by_budget else d.groupby(level="seed").mean()


def level_by_budget(frame: pd.DataFrame, split: str, metric: str, cell) -> pd.DataFrame:
    """Per seed and budget: mean over draws of the metric of one cell (rows seed, columns budget)."""
    g = frame[(frame.split == split) & (frame.selector == cell[0]) & (frame.learner == cell[1])]
    return g.groupby(["seed", "budget"])[metric].mean().unstack("budget")


def boot_ci(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    b = np.random.default_rng(seed).choice(x, size=(n, len(x))).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def wilcoxon_p(x) -> float:
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or not np.any(x != 0): return 1.0
    try: return float(wilcoxon(x).pvalue)
    except ValueError: return 1.0


def summarize(x, n=N_BOOT) -> dict:
    x = pd.Series(x).dropna(); lo, hi = boot_ci(x.to_numpy(), n)
    return {"gain": float(x.mean()), "lo": lo, "hi": hi, "share_better": float((x > 0).mean()), "p": wilcoxon_p(x), "n": int(len(x))}


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def fmt_gain(g, lo=None, hi=None, d=3) -> str:
    s = f"{g:+.{d}f}"
    return s if lo is None else f"{s} [{lo:+.{d}f}, {hi:+.{d}f}]"


def fmt_p(p) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.1 else f"{p:.3f}"


def to_md(df: pd.DataFrame, floatfmt="{:+.4f}") -> str:
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out) + "\n"


class Numbers:
    """Number store: scripts register every number that the prose may quote under a stable key; tables/numbers.json is the merge of all stores (written per script to avoid races)."""
    def __init__(self, name: str): self.name = name; self.d = {}
    def put(self, key: str, value):
        if isinstance(value, (np.floating, np.integer)): value = value.item()
        if isinstance(value, float) and np.isnan(value): value = None
        assert key not in self.d, f"duplicate number key {key}"; self.d[key] = value
    def save(self):
        TABLES.mkdir(exist_ok=True); (TABLES / f"numbers_{self.name}.json").write_text(json.dumps(self.d, indent=1, sort_keys=True))


def all_numbers() -> dict:
    out = {}
    for f in sorted(TABLES.glob("numbers_*.json")):
        d = json.loads(f.read_text()); dup = set(out) & set(d); assert not dup, f"duplicate keys across number stores: {sorted(dup)[:5]}"; out.update(d)
    return out


def write(path: Path, text: str):
    path.parent.mkdir(exist_ok=True, parents=True); path.write_text(text); print("wrote", path.relative_to(PKG))

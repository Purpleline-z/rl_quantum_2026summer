"""Per-type view of the harsh-split gain from the saved per-judgment scores: gain of lap3_k10 over aw8_all on the HTR judgments (type 4) and on all other types, pooled over seeds, budgets and draws, with a test-group bootstrap.
Descriptive: the HTR test judgments are very few (see counts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import ge_analyze as ga

R = ga.R


def per_type(block: str, kind: str, reps: int = 300, seed: int = 0) -> dict:
    files = sorted((R / f"{block}_harsh").glob(f"seed*_{kind}.d.npz")); z0 = np.load(files[0], allow_pickle=True); groups = z0["meta_group"]; k = z0["meta_k"]; uniq = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in uniq}
    data = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        for key in {key[2:].rsplit("_", 1)[0] for key in z.files if key.startswith("d_")}:
            if f"d_{key}_lap3" in z.files and f"d_{key}_base" in z.files: data.append((z[f"d_{key}_lap3"], z[f"d_{key}_base"]))
    s, w = z0["meta_sign"], z0["meta_weight"]
    def acc(d, sel): return float((w[sel] * ((d[sel] * s[sel]) > 0)).sum() / w[sel].sum())
    def gain(sel): return float(np.mean([acc(a, sel) - acc(b, sel) for a, b in data]))
    out = {}
    for name, mask in (("HTR", k == 4), ("other types", k != 4)):
        base = gain(np.flatnonzero(mask)); rng = np.random.default_rng(seed); boots = []
        for _ in range(reps):
            pick = rng.choice(uniq, len(uniq)); sel = np.concatenate([idx[g] for g in pick]); sel = sel[mask[sel]]
            boots.append(gain(sel) if len(sel) else np.nan)
        out[name] = (base, *np.nanpercentile(boots, [2.5, 97.5]), int(mask.sum()))
    return out


def table() -> pd.DataFrame:
    rows = []
    for block in ("conf1", "conf2"):
        for kind in ("run", "temp"):
            if not list((R / f"{block}_harsh").glob(f"seed*_{kind}.d.npz")): continue
            for name, (g, lo, hi, n) in per_type(block, kind).items(): rows.append({"block": block, "kind": kind, "types": name, "test_judgments": n, "acc_gain": g, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(ga.md(table()))

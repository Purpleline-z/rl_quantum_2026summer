"""Offline fusion analyses from the STORED per-judgment scores (no model is refitted): how would a per-type route / blend of the head with a free-score model have done?

  python3 sp_offline.py scan DIR [DIR ...]                  route HTR to c * d_free for every stored free-score variant and c in {1, 2, 3}, ranked by the mean of the four AUC/accuracy gains over the head
  python3 sp_offline.py strata DIR [DIR ...] --fs NAME --c 2   gains of the HTR route by anchor-distance stratum and for the HTR judgments themselves

All numbers are development observations when run on development seeds (the choice of the type to route was made after looking at per-type tables of the same seeds).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import rc_tables as T

HEAD = "lap3_k10_st4"


def load(dirs, names=None):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "seed*.json"))):
            b = json.load(open(f)); m = b["meta"]
            for r in b["records"]:
                if names is None or r["learner"] in names: out[(r["seed"], r["split"], r["budget"], r["draw"], r["learner"])] = (np.array(r["d"]), m)
    return out


def metr(d, s, w, mask):
    s, w, d = s[mask], w[mask], d[mask]
    if mask.sum() < 4: return np.nan, np.nan
    return (roc_auc_score(s > 0, d) if len(set(s)) == 2 else np.nan), float((w * ((d * s) > 0)).sum() / w.sum())


def route(dh, df, k, c, type_idx=4): return np.where(k == type_idx, c * df, dh)


def gains(D, fsn, comb, mask_fn=None):
    cells = sorted({k[:4] for k in D if k[4] == fsn and (*k[:4], HEAD) in D}); res = {"A": {}, "B": {}}
    for cell in cells:
        dh, m = D[(*cell, HEAD)]; df, _ = D[(*cell, fsn)]; k, s, w, anc = (np.array(m[x]) for x in ("k", "sign", "weight", "anc_min")); mask = np.ones(len(k), bool) if mask_fn is None else mask_fn(k, anc)
        a, b = metr(dh, s, w, mask), metr(comb(dh, df, k), s, w, mask); res[cell[1]].setdefault(cell[0], []).append((b[0] - a[0], b[1] - a[1]))
    if not all(res.values()): return None
    return [np.nanmean([np.nanmean(np.array(v), 0) for v in res[sp].values()], 0) for sp in "AB"], len(res["A"])


def scan(dirs) -> pd.DataFrame:
    D = load(dirs); fss = sorted({k[4] for k in D if k[4].split("_")[0] in ("fs", "kn", "ls")}); rows = []
    for fsn in fss:
        for c in (1, 2, 3):
            g = gains(D, fsn, lambda dh, df, k, c=c: route(dh, df, k, c))
            if g is None: continue
            o, n = g; rows.append({"free-score variant": fsn, "c": c, "n_seeds": n, "auc_A": o[0][0], "auc_B": o[1][0], "acc_A": o[0][1], "acc_B": o[1][1], "score": np.mean([o[0][0], o[1][0], o[0][1], o[1][1]])})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def strata(dirs, fsn, c) -> pd.DataFrame:
    D = load(dirs); rows = []
    for name, fn in (("all judgments", None), ("near (<=0.001)", lambda k, a: a <= 0.001), ("0.001-0.03", lambda k, a: (a > 0.001) & (a <= 0.03)), ("far (>0.03)", lambda k, a: a > 0.03), ("HTR judgments only", lambda k, a: k == 4), ("non-HTR judgments", lambda k, a: k != 4)):
        g = gains(D, fsn, lambda dh, df, k: route(dh, df, k, c), fn)
        if g is None: continue
        o, n = g; rows.append({"stratum": name, "n_seeds": n, "auc_A": o[0][0], "auc_B": o[1][0], "acc_A": o[0][1], "acc_B": o[1][1]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("mode", choices=["scan", "strata"]); ap.add_argument("dirs", nargs="+"); ap.add_argument("--fs", default="fs_lam10_rho0.001"); ap.add_argument("--c", type=float, default=2.0)
    a = ap.parse_args(); dirs = [str((T.NEW / d) if not Path(d).exists() else d) for d in a.dirs]
    print(T.md(scan(dirs).head(20)) if a.mode == "scan" else T.md(strata(dirs, a.fs, a.c)))


if __name__ == "__main__": main()

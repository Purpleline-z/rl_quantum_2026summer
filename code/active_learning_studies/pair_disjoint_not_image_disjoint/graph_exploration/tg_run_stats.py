"""Descriptive statistics of the three growth runs from the file names (<index>_<run>_<temperature>C_<frame>[-k].bmp): is temperature monotone in time? how far apart are consecutive indices in camera frames?
  python3 tg_run_stats.py [DATA_DIR]       (pure pandas; script-generated table for the results document)"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tg_tables import md

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3].parent / "data" / "original data" / "Trajectories"
rows = []
for f in root.glob("*/*.bmp"):
    m = re.match(r"(\d+)_(RR\d+[A-Z]?)_(\d+)C?_(\d+)(?:-(\d+))?\.bmp$", f.name)
    if m: rows.append(dict(idx=int(m[1]), run=m[2], temp=int(m[3]), frame=int(m[4])))
d = pd.DataFrame(rows).sort_values(["run", "idx"]); out = []
for run, g in d.groupby("run"):
    dT = np.diff(g.temp.to_numpy()); dF = np.diff(g.frame.to_numpy()); rho = spearmanr(g.idx, g.temp)[0]
    out.append({"run": run, "frames": len(g), "temperature first -> last (C)": f"{g.temp.iloc[0]} -> {g.temp.iloc[-1]}", "Spearman(index, temperature)": rho, "steps with dT>0": float((dT > 0).mean()), "steps with dT<0": float((dT < 0).mean()),
                "steps with |dT|>20 C": float((np.abs(dT) > 20).mean()), "median camera-frame gap between consecutive indices": float(np.median(dF)), "p90 gap": float(np.percentile(dF, 90)), "temperature sign changes along the run": int((np.diff(np.sign(dT[dT != 0])) != 0).sum())})
t = pd.DataFrame(out); t["frames"] = t["frames"].astype(str); print(md(t, "{:.3f}"))

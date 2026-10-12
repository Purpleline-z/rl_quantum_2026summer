"""Across all cells (single-shot, both splits), relate a selector's log-loss gain over Random to the decisive share of the judgments it revealed.
usage: python3 share_vs_gain.py   -> prints correlation and a binned table; also AUC and calibrated-LL gains for comparison."""
import json, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2] / "results" / "judgment_unit_study"
rows = []
for split in "AB":
    base = ROOT / f"{split}_groups"; man = {}
    for f in (base).glob("manifest_seed*.json"): man[int(re.search(r"seed(\d+)", f.name).group(1))] = json.loads(f.read_text())
    for cond in ("single", "sequential"):
        for f in (base / cond).glob("seed*_*.json"):
            d = json.loads(f.read_text()); seed = d["seed"]
            if "checkpoints" not in d or seed not in man: continue
            pool = man[seed]["pool"]
            for b, v in d["checkpoints"].items():
                sel = v["selected"]; share = np.mean([pool[i][3] in ("1", "2") for i in sel]); rows.append(dict(split=split, cond=cond, seed=seed, strategy=d["strategy"], budget=int(b), share=share, ll=v["test_decisive_log_loss"], auc=v["test_decisive_auc"], acc=v["test_decisive_accuracy"]))
df = pd.DataFrame(rows); df["family"] = df.strategy.str.replace(r"^random_r\d+$", "random", regex=True)
g = df.groupby(["split", "cond", "seed", "budget", "family"], as_index=False)[["share", "ll", "auc", "acc"]].mean()
rnd = g[g.family == "random"].drop(columns="family").rename(columns={"share": "r_share", "ll": "r_ll", "auc": "r_auc", "acc": "r_acc"}); m = g[g.family != "random"].merge(rnd, on=["split", "cond", "seed", "budget"])
m["d_share"] = m.share - m.r_share; m["gain_ll"] = m.r_ll - m.ll; m["gain_auc"] = m.auc - m.r_auc; m["gain_acc"] = m.acc - m.r_acc
print("cells", len(m), "strategies", m.family.nunique(), "mean decisive share of Random", round(rnd.r_share.mean(), 3))
for s in "AB":
    for c in ("single", "sequential"):
        x = m[(m.split == s) & (m.cond == c)]
        print(f"{s}-{c[:3]}: Spearman(d_share, gain_ll) = {spearmanr(x.d_share, x.gain_ll)[0]:+.2f}   (d_share, gain_auc) = {spearmanr(x.d_share, x.gain_auc)[0]:+.2f}   (d_share, gain_acc) = {spearmanr(x.d_share, x.gain_acc)[0]:+.2f}   n={len(x)}")
m["bin"] = pd.cut(m.d_share, [-1, -.15, -.05, .05, .15, 1]); print("\nlog-loss / AUC / accuracy gain over Random by difference in decisive share (selector minus Random):")
print(m.groupby("bin", observed=True)[["gain_ll", "gain_auc", "gain_acc"]].agg(["mean", "count"]).round(3).to_string())
m.to_csv(Path(__file__).parent / "share_vs_gain_cells.csv", index=False)

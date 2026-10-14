"""Summary figure of the HTR failure analysis (results/hf/hf_summary.png): per-judgment error rate of lap3_k10_st4 and the development gains of the variants under the standard and anchor-strict endpoints."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nd_core as core

HF = core.HERE / "results" / "hf"
pj = pd.read_csv(HF / "errors_dev_per_judgment.csv").sort_values("err_rate", ascending=False).reset_index(drop=True); c = pd.read_csv(HF / "dev_selection_conditions.csv")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
colors = {"HTR-RT13": "#1b6ca8", "1x1-HTR": "#2a9d8f"}; cols = [colors.get(k, "#c0392b") for k in pj.cls_pair]
ax[0].bar(range(len(pj)), pj.err_rate, color=cols); ax[0].set_xticks(range(len(pj))); ax[0].set_xticklabels(pj.jid, rotation=90, fontsize=7); ax[0].set_ylabel("error rate of lap3_k10_st4 on this judgment\n(pooled over seeds, splits, budgets, draws)"); ax[0].set_xlabel("decisive HTR judgment (id)")
ax[0].set_title("45 decisive HTR judgments: two carry all the errors"); ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in ("#1b6ca8", "#2a9d8f", "#c0392b")], labels=["HTR frame vs RT13-like frame", "HTR frame vs 1x1-like frame", "neither frame is HTR-like (nearest ideal class)"], fontsize=8, loc="upper right")
order = ["hfb_g5", "hfb_g10", "hfb_g20", "hfb3_g10", "hfs", "hfq1", "hfc1", "hfcb10"]; w = 0.2
for i, (e, s, col) in enumerate((("standard", "A", "#1b6ca8"), ("standard", "B", "#7fb2d9"), ("strict0.01", "A", "#c0392b"), ("strict0.01", "B", "#e8a29b"))):
    v = [float(c[(c.cand == k) & (c.endpoint == e) & (c.split == s)].ll_HTR.iloc[0]) for k in order]; ax[1].bar(np.arange(len(order)) + (i - 1.5) * w, v, w, color=col, label=f"{e} split {s}")
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=30, fontsize=8); ax[1].set_ylabel("HTR log-loss gain over lap3_k10_st4 (dev, 20 seeds)"); ax[1].legend(fontsize=8); ax[1].set_title("Variant gains vanish or reverse when near-copy anchors are removed")
fig.tight_layout(); fig.savefig(HF / "hf_summary.png", dpi=110); print("saved")

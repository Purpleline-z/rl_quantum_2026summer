"""Reliability diagrams of the calibration study (script-generated from the stage-1 cells).

  python3 cal_figures.py SPEC.json OUTDIR DIR [DIR...]      (SPEC as for cal_tables.py confirm; draws raw + the candidates)
Panels: (1) per split, all judgments; (2) by type (both splits pooled); (3) anchor-far vs anchor-near (both splits pooled).  Equal-width confidence bins on [0.5, 1]; x = mean predicted confidence of the
predicted winner, y = observed share of correct predictions (confidence-weighted); the dot area is proportional to the weight in the bin.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd_calibration as C, cal_tables as T

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def binned(d, sign, w, mask, nb=10):
    p = 1 / (1 + np.exp(-d[mask])); conf = np.maximum(p, 1 - p); ok = ((p > .5) == (sign[mask] > 0)).astype(float); b = np.clip(((conf - .5) / .5 * nb).astype(int), 0, nb - 1); ww = w[mask]
    return (np.bincount(b, ww * conf, nb), np.bincount(b, ww * ok, nb), np.bincount(b, ww, nb))


def accumulate(cells, names, priors, groups):
    acc = {(g, n): [np.zeros(10)] * 3 for g in groups for n in names}
    for cell in cells:
        pri = priors
        if priors == "loo": pri = {"type": C.fit_prior([r for c in cells if c["seed"] != cell["seed"] for r in c["recs"].values()])}
        for rec in cell["recs"].values():
            v = C.variants(dict(rec), pri); t = rec["test"]
            for n in names:
                d = np.asarray(v[n], dtype=float)
                for g, mask_fn in groups.items():
                    m = mask_fn(cell, t)
                    if m.any(): acc[(g, n)] = [a + b for a, b in zip(acc[(g, n)], binned(d, t["sign"], t["weight"], m))]
    return acc


def draw(ax, acc, group, names, title):
    ax.plot([.5, 1], [.5, 1], color="#999999", lw=1, ls="--")
    for i, n in enumerate(names):
        c, a, w = acc[(group, n)]; ok = w > 0; ax.plot((c / np.where(ok, w, 1))[ok], (a / np.where(ok, w, 1))[ok], "-", color=COLORS[i % 6], lw=1.2, label=n); ax.scatter((c / np.where(ok, w, 1))[ok], (a / np.where(ok, w, 1))[ok], s=250 * w[ok] / max(w.sum(), 1), color=COLORS[i % 6], alpha=.6)
    ax.set_title(title, fontsize=9); ax.set_xlabel("mean confidence"); ax.set_ylabel("observed accuracy"); ax.set_xlim(.5, 1.01); ax.set_ylim(.4, 1.01)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); cells = T.load_cells(sys.argv[3:]); names = [spec["reference"]] + spec["candidates"]
    pr = spec.get("priors"); priors = {"type": {int(k): v for k, v in pr["type"].items()}} if pr else ("loo" if spec.get("loo_priors") else None)
    groups = {f"split{s}": (lambda cell, t, s=s: np.full(len(t["sign"]), cell["split"] == s)) for s in "AB"}
    groups.update({f"type_{name}": (lambda cell, t, k=k: t["k"] == k) for k, name in C.TYPE_NAME.items()})
    groups.update({"far": lambda cell, t: t["g_anc_min"] > C.FAR_THRESHOLD, "near": lambda cell, t: t["g_anc_min"] <= C.FAR_THRESHOLD})
    acc = accumulate(cells, names, priors, groups)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4)); [draw(axes[i], acc, f"split{s}", names, f"split {s}: all decisive test judgments") for i, s in enumerate("AB")]; axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "reliability_splits.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8)); [draw(axes[i], acc, f"type_{name}", names, f"type {name}") for i, name in enumerate(C.TYPE_NAME.values())]; axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "reliability_types.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4)); draw(axes[0], acc, "near", names, "anchor-near (some image <= 0.03 from an anchor)"); draw(axes[1], acc, "far", names, "anchor-far (both images > 0.03)"); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "reliability_anchor.png", dpi=130); plt.close(fig)
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__": main()

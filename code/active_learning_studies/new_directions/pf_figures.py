"""Figures for the physicists (written to graph_exploration/figures_pf/):  (1) detection overlays (horizon, specular spot, band boundaries) for sample ideal images of every type, (2) mean spacing-normalised profiles by type
and band with spread, (3) PCA of the physics descriptors by type (8-bit and 16-bit images marked).   python3 pf_figures.py"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import nd_core as core, pf_extract as P

OUT = core.HERE.parent / "pair_disjoint_not_image_disjoint" / "graph_exploration" / "figures_pf"; TYPES = {"(1x1)": "STO_ideal_1x1", "c(6x2)": "STO_ideal_c6x2", "RT13": "STO_ideal_RT13", "HTR": "STO_ideal_HTR"}; COL = {"(1x1)": "#4C78A8", "c(6x2)": "#F58518", "RT13": "#54A24B", "HTR": "#E45756"}


def overlays(ex):
    paths = sorted(ex); tiles = []; rng = np.random.default_rng(3)
    for t, k in TYPES.items():
        ps = [p for p in paths if k in p and ex[p]["mode"] != "I;16"]; pick = rng.choice(ps, 3, replace=False)
        for p in pick:
            g, _ = P.load_gray(p); e = ex[p]; im = Image.fromarray((np.clip(g / 255, 0, 1) ** .5 * 255).astype(np.uint8)).convert("RGB"); d = ImageDraw.Draw(im); d.line([(0, e["edge"]), (656, e["edge"])], fill=(255, 0, 0), width=2)
            r, x = e["row"], e["col"]; d.ellipse([x - 9, r - 9, x + 9, r + 9], outline=(0, 255, 0), width=2)
            for b, (a0, a1) in P.BANDS.items(): d.rectangle([x - P.HALF, max(r + a0, e["edge"] + 3), x + P.HALF, max(r + a1, e["edge"] + 4)], outline=(80, 160, 255))
            d.text((6, 6), t, fill=(255, 255, 0)); tiles.append(im.resize((328, 246)))
    W = Image.new("RGB", (328 * 3, 246 * 4))
    for i, t in enumerate(tiles): W.paste(t, ((i % 3) * 328, (i // 3) * 246))
    W.save(OUT / "detection_overlays.png")


def profiles(ex):
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8)); paths = sorted(ex); grid = np.arange(0.05, 3.0, 0.05)
    for ax, b in zip(axs, "abcd"):
        for t, k in TYPES.items():
            rows = []
            for p in paths:
                if k in p and ex[p]["mode"] != "I;16":
                    q = P.baseline_corrected(ex[p]["profiles"][b]); q = q / (np.percentile(q, 99.5) + 1e-6); rows.append(P._interp_u(q, P.spacing(ex[p]), grid, 0.03))
            m, s = np.mean(rows, 0), np.std(rows, 0); ax.plot(grid, m, color=COL[t], label=f"{t} (n={len(rows)})"); ax.fill_between(grid, np.clip(m - s, 0, None), m + s, color=COL[t], alpha=.12)
        ax.set_title(f"band {b}: rows {P.BANDS[b][0]}..{P.BANDS[b][1]} px from the specular row"); ax.set_xlabel("distance from specular rod / integer spacing a")
        for u in (1 / 3, 1 / 2, 2 / 3, 1.0): ax.axvline(u, color="0.8", lw=.6)
    axs[0].legend(fontsize=7); axs[0].set_ylabel("normalised intensity (8-bit ideals)"); plt.tight_layout(); plt.savefig(OUT / "profiles_by_type.png", dpi=80); plt.close()


def pca_plot(ex):
    b = torch.load(core.HERE / "results" / "pf" / "blocks_v1.pt", weights_only=False); paths = b["paths"]; X = np.hstack([b["blocks"]["pxprof"], b["blocks"]["bg"], b["blocks"]["width"]]); tr = [i for i, p in enumerate(paths) if "/Trajectories/" in p]
    Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6); Z = np.clip(Z, -5, 5); u, s, vt = np.linalg.svd(Z[tr] - Z[tr].mean(0), full_matrices=False); Y = (Z - Z[tr].mean(0)) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.scatter(Y[tr, 0], Y[tr, 1], s=4, c="0.85", label="trajectory frames")
    for t, k in TYPES.items():
        for is16, mk in ((False, "o"), (True, "^")):
            idx = [i for i, p in enumerate(paths) if k in p and (ex[p]["mode"] == "I;16") == is16]
            if idx: ax.scatter(Y[idx, 0], Y[idx, 1], s=26, marker=mk, color=COL[t], edgecolor="k", linewidth=.3, label=f"{t}{' (16-bit file)' if is16 else ''}")
    ax.legend(fontsize=6); ax.set_xlabel("PC 1 of the physics descriptors (pfa)"); ax.set_ylabel("PC 2"); plt.tight_layout(); plt.savefig(OUT / "pca_physics.png", dpi=80)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); ex = torch.load(core.HERE / "results" / "pf" / "extract.pt", weights_only=False); overlays(ex); profiles(ex); pca_plot(ex); print("figures written to", OUT)

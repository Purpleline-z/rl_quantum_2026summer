"""Physics-informed RHEED feature extractor (claude/physics-features-*): label-free, no learning, raw pixels only.  See PHYSICS_FEATURES.md for the physicist-facing description.

Geometry of a RHEED pattern (all images: electron beam grazing on the sample, screen with the specular spot, streaks = intersections of reciprocal-lattice rods with the Ewald sphere):
  * the shadow edge (sample horizon) is the row where the (dark) region above becomes bright (diffuse background);
  * the specular spot S is the brightest compact blob below the edge; its column x0 is the centre of the pattern (the (0,0) rod);
  * integer-order rods stand at +-a, +-2a ... (a = integer-order spacing in pixels), fractional-order rods (reconstructions) at fractions of a;
  * RT13 / HTR / c(6x2) differ by which fractional positions carry intensity, their relative heights, their widths and the spot arrays below the specular row.
Steps (``extract``): decode (8-bit palette images via PIL 'L'; 16-bit PNGs divided by 257 so that the brightness scale equals that of the 8-bit images; 16-bit images are 780x580 and are resampled by 656/780 to the
common 656x492 frame) -> shadow edge -> specular spot -> horizontal intensity profiles of several vertical bands relative to S, background-subtracted and amplitude-normalised -> descriptors.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

W0, H0 = 656, 492


def load_gray(path) -> tuple[np.ndarray, str]:
    """Float image in [0, 255] on the common 656x492 frame; also the original PIL mode."""
    im = Image.open(path); mode = im.mode
    if mode == "I;16": a = np.asarray(im, dtype=np.float32) / 257.0; a = np.asarray(Image.fromarray(a).resize((W0, H0), Image.BILINEAR), dtype=np.float32)
    else: a = np.asarray(im.convert("L"), dtype=np.float32)
    return a, mode


HORIZON_HI = 0.40      # search window of the horizon: 5 % ... HORIZON_HI of the height (v1 of the cache used 0.70, which mislocated the horizon at the specular band in ~7 % of the trajectory frames; v2 uses 0.40)


def shadow_edge(g: np.ndarray, hi_frac: float | None = None) -> int:
    """Row of the sample horizon: strongest upward-to-downward brightening of the (outlier-robust) row level of the central columns, searched between 5 % and ``hi_frac`` (default 40 %) of the height."""
    h, w = g.shape; lvl = np.median(g[:, int(.25 * w):int(.75 * w)], axis=1); lvl = ndi.gaussian_filter1d(lvl, 3); d = np.gradient(lvl); lo, hi = int(.05 * h), int((hi_frac or HORIZON_HI) * h)
    return int(lo + np.argmax(d[lo:hi]))


def specular(g: np.ndarray, edge: int) -> tuple[int, int]:
    """(row, col) of the specular spot: centroid of the brightest compact blob of the lightly smoothed image below the horizon (central 60 % of the columns, top 55 % of the rows below the edge)."""
    h, w = g.shape; s = ndi.gaussian_filter(g, 4.0)
    r0, r1, c0, c1 = min(edge + 4, h - 2), min(edge + int(.55 * h), h), int(.2 * w), int(.8 * w); sub = s[r0:r1, c0:c1]; i = np.unravel_index(np.argmax(sub), sub.shape)
    pr, pc = r0 + i[0], c0 + i[1]; y0, y1, x0, x1 = max(pr - 15, 0), min(pr + 16, h), max(pc - 15, 0), min(pc + 16, w); win = s[y0:y1, x0:x1]; m = win > .6 * win.max(); wgt = (win - .6 * win.max()) * m
    yy, xx = np.mgrid[y0:y1, x0:x1]; return int(round((wgt * yy).sum() / wgt.sum())), int(round((wgt * xx).sum() / wgt.sum()))


def profile(g: np.ndarray, r0: int, r1: int, x0: int, halfwidth: int = 260) -> np.ndarray:
    """Mean intensity of rows [r0, r1) as a function of the column offset from x0 (offsets -halfwidth..halfwidth, NaN outside the frame)."""
    h, w = g.shape; r0, r1 = max(0, r0), min(h, max(r1, r0 + 1)); p = g[r0:r1].mean(0); out = np.full(2 * halfwidth + 1, np.nan, dtype=np.float32)
    lo, hi = max(0, x0 - halfwidth), min(w, x0 + halfwidth + 1); out[lo - (x0 - halfwidth):hi - (x0 - halfwidth)] = p[lo:hi]; return out


# ------------------------------------------------------------------------------------------ band profiles and descriptors
HALF = 240                                   # profile half width in pixels (656-px frame)
BANDS = {"a": (-45, -10), "b": (-10, 10), "c": (15, 60), "d": (60, 130), "e": (130, 230)}   # row ranges relative to the specular row (pixels)
BIN = 8                                      # folded-profile bin width in pixels -> HALF // BIN = 30 bins per band
EPS = 1e-6


def baseline_corrected(p: np.ndarray) -> np.ndarray:
    """Profile minus a slowly varying baseline (20th percentile in a 61-px window = diffuse background, Kikuchi lines, brightness gradient); negative values clipped."""
    p = np.nan_to_num(p, nan=float(np.nanmedian(p)) if np.isfinite(p).any() else 0.0); base = ndi.percentile_filter(p, 20, size=61, mode="nearest"); return np.clip(p - base, 0, None)


def fold(q: np.ndarray, bin_px: int = BIN) -> np.ndarray:
    """Left-right symmetrised profile |x| = 0..HALF averaged in bins of ``bin_px`` pixels (the reciprocal-lattice pattern is mirror-symmetric about the specular rod)."""
    s = (q[HALF:] + q[HALF::-1]) / 2; n = HALF // bin_px; return s[:n * bin_px].reshape(n, bin_px).mean(1)


def extract(path, hi_frac: float | None = None) -> dict:
    g, mode = load_gray(path); h, w = g.shape; edge = shadow_edge(g, hi_frac); r, x0 = specular(g, edge); out = {"mode": mode, "edge": edge, "row": r, "col": x0}
    out["lvl_above"] = float(g[:max(edge - 5, 1)].mean()); out["lvl_below"] = float(g[edge + 5:edge + 120].mean()); prof = {}
    for b, (a0, a1) in BANDS.items():
        lo, hi = max(r + a0, edge + 3), r + a1                       # rows must lie below the horizon
        p = profile(g, lo, hi, x0, HALF) if hi > lo else np.full(2 * HALF + 1, np.nan, dtype=np.float32); prof[b] = p
    out["profiles"] = prof; return out


# ------------------------------------------------------------------------------------------ descriptor blocks
def spacing(ex: dict, lo: int = 25, hi: int = 110) -> float:
    """Integer-order spacing a (pixels): position of the strongest peak of the folded, baseline-corrected band-a profile (rows just above the specular row) in [lo, hi] pixels."""
    q = baseline_corrected(ex["profiles"]["a"]); s = (q[HALF:] + q[HALF::-1]) / 2; s = ndi.gaussian_filter1d(s, 1.5); return float(lo + np.argmax(s[lo:hi + 1]))


def _interp_u(q: np.ndarray, a: float, grid: np.ndarray, win: float = 0.03) -> np.ndarray:
    """Folded profile sampled at positions u*a (u in ``grid``): maximum over u +- win (peak height tolerant to small shifts)."""
    s = (q[HALF:] + q[HALF::-1]) / 2; out = np.zeros(len(grid), dtype=np.float32); n = len(s)
    for i, u in enumerate(grid):
        lo, hi = int(max(0, np.floor((u - win) * a))), int(min(n - 1, np.ceil((u + win) * a))); out[i] = s[lo:hi + 1].max() if hi >= lo else 0.0
    return out


FRACS = np.array([1 / 13, 2 / 13, 3 / 13, 1 / 6, 1 / 4, 1 / 3, 4 / 13, 5 / 13, 1 / 2, 6 / 13, 7 / 13, 2 / 3, 3 / 4, 5 / 6, 1.0, 7 / 6, 4 / 3, 1.5, 5 / 3, 2.0, 2.5, 3.0])   # candidate rod positions in units of the integer spacing
UGRID = np.round(np.arange(0.1, 3.01, 0.1), 3)


def descriptors(ex: dict) -> dict[str, np.ndarray]:
    """All descriptor blocks of one image (see PHYSICS_FEATURES.md)."""
    a = spacing(ex); prof = ex["profiles"]; q = {b: baseline_corrected(prof[b]) for b in prof}; A = {b: np.percentile(q[b], 99.5) + EPS for b in q}; n = {b: q[b] / A[b] for b in q}
    out = {"spacing": np.array([a / 62.0], dtype=np.float32)}
    out["pxprof"] = np.concatenate([fold(n[b], BIN) for b in "abcd"]).astype(np.float32)                                   # 4 bands x 30 bins of 8 px (pixel units)
    out["uprof"] = np.concatenate([_interp_u(n[b], a, UGRID) for b in "abcd"]).astype(np.float32)                        # 4 bands x 30 points in units of the integer spacing
    spec = max(float(q["b"][HALF - 6:HALF + 7].max()), EPS)                                                               # specular-rod peak (band b)
    pk = []
    for b in "abcd":
        h = _interp_u(q[b], a, FRACS, 0.04); pk.append(h / (h[FRACS == 1.0][0] + EPS) if b != "b" else h / spec)         # fractional-order peak heights relative to the first-order rod (band b: relative to the specular rod)
    out["peaks"] = np.clip(np.concatenate(pk), 0, 5).astype(np.float32)
    f = []
    for b in "abd":                                                                                                       # amplitude of the Fourier component of the profile at candidate periods of a/m
        x = n[b] - n[b].mean(); xs = np.arange(len(x)) - HALF; wnd = np.hanning(len(x)); fr = []
        for m in (1, 2, 3, 6, 13 / 2, 13 / 3, 13):
            ph = np.exp(-2j * np.pi * m * xs / a); fr.append(abs((x * wnd * ph).sum()) / wnd.sum())
        f += fr
    out["fourier"] = np.array(f, dtype=np.float32)
    sig = {b: float(q[b].mean()) for b in q}; out["bg"] = np.array([np.log(A[b]) for b in "abcde"] + [np.log(sig[b] + EPS) for b in "abcde"] + [np.log(ex["lvl_above"] + 1), np.log(ex["lvl_below"] + 1), (ex["row"] - ex["edge"]) / 100.0], dtype=np.float32)
    w = n["b"][HALF - 40:HALF + 41]; out["width"] = np.array([float((w > .5).sum()) / 10, float(w.sum()) / 10, float(n["b"][HALF - 6:HALF + 7].max())], dtype=np.float32)   # specular-rod width descriptors
    return out

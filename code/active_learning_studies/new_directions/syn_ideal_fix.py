"""Real-image preprocessing controls for the typed anchors (no simulator).

Findings behind it (script-generated table: ``python3 syn_ideal_fix.py --report``):
  * 18 of the 154 ideal images (10 HTR, 8 RT13) are 16-bit PNGs (780 x 580); the project's loader does ``Image.open(p).convert("L")``, which CLIPS 16-bit values at 255, so ~88 % of the pixels are saturated and
    their features are almost constant (~0.37 cosine distance from every trajectory frame).
  * the ideal images of (1x1) and c(6x2) are uncropped frames that show the direct beam + halo above the shadow edge, trajectory frames do not; the direct beam dominates the SimCLR feature of such
    an ideal image (median distance to the nearest trajectory frame 0.69 for (1x1)); masking the rows above the shadow edge reduces it to ~0.17.
Variants of the ideal-image features (path -> 512-d feature), cached in results/syn/ideal_fixed.pt:
  fix16      16-bit PNGs rescaled by 255/65535 instead of clipped (the other images unchanged)
  mask       fix16 + rows above the shadow edge (135/492 of the height) set to 0 in EVERY ideal image
  mask_keep16  rows above the shadow edge set to 0 in the 8-bit ideal images; the 16-bit PNGs keep their loader (clipped) features
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
import nd_core as core
from syn_bank import encoder_and_tf, encode_images, OUT

FILE = OUT / "ideal_fixed.pt"; SHADOW = 135 / 492


def load_fixed(p, mask: bool):
    im = Image.open(p); a = np.array(im).astype(np.float64)
    if a.max() > 255: a = a / 65535.0 * 255.0
    if mask: a = a.copy(); a[: int(round(SHADOW * a.shape[0]))] = 0
    return np.clip(a, 0, 255).astype(np.uint8)


def build():
    enc, tf, cache = encoder_and_tf(); paths = sorted(p for p in cache if "STO_ideal" in p); out = {"paths": paths}
    for name, mask in (("fix16", False), ("mask", True)):
        feats = encode_images(enc, tf, [load_fixed(p, mask) for p in paths])
        if name == "fix16":   # only the 16-bit files change
            keep = torch.stack([cache[p].float() for p in paths]); is16 = torch.tensor([Image.open(p).mode.startswith("I;16") for p in paths]); feats = torch.where(is16[:, None], feats, keep)
        out[name] = feats
    keep = torch.stack([cache[p].float() for p in paths]); is16 = torch.tensor([Image.open(p).mode.startswith("I;16") for p in paths]); out["mask_keep16"] = torch.where(is16[:, None], keep, out["mask"])
    OUT.mkdir(parents=True, exist_ok=True); torch.save(out, FILE); return out


def load():
    return torch.load(FILE, weights_only=False) if FILE.exists() else build()


if __name__ == "__main__":
    torch.set_num_threads(4); d = build(); print(FILE, {k: tuple(v.shape) for k, v in d.items() if k != "paths"})

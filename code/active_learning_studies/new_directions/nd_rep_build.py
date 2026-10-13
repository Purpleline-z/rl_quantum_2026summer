"""Frozen-encoder representations for the representation study (claude/representation-*): everything label-free and CPU.

  python3 nd_rep_build.py views     SimCLR final feature of 7 deterministic views per image (original, horizontal mirror, centre crop and 4 corner crops of 90 % side, resized to 224) -> results/rep/simclr_views.pt
  python3 nd_rep_build.py layers    (reuses nd_gm_layers.build if results/feature_layers_all.pt is missing) pooled layer2/3/4 outputs
Other encoder: the ImageNet ResNet-18 features of image_representation_analysis (1278 images, cached; its weights are NOT available offline: HuggingFace and download.pytorch.org answer 403 through the proxy)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import nd_core as core

REP_DIR = core.HERE / "results" / "rep"; VIEWS = ("orig", "flip", "ce", "tl", "tr", "bl", "br")


def build_views() -> None:
    cache = core.feature_cache(); torch.set_num_threads(4); REP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(8600, "A", scratch, cache); enc = ctx.features.base_model.encoder.eval(); tf = ctx.features.tf; paths = sorted(cache); out = {}
        def views(img):
            w, h = img.size; cw, ch = int(round(.9 * w)), int(round(.9 * h)); ox, oy = (w - cw) // 2, (h - ch) // 2
            return [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.crop((ox, oy, ox + cw, oy + ch)), img.crop((0, 0, cw, ch)), img.crop((w - cw, 0, w, ch)), img.crop((0, h - ch, cw, h)), img.crop((w - cw, h - ch, w, h))]
        with torch.no_grad():
            for i in range(0, len(paths), 8):
                chunk = paths[i:i + 8]; batch = torch.stack([tf(v) for p in chunk for v in views(Image.open(p).convert("L"))]); z = enc(batch).reshape(len(chunk), len(VIEWS), -1)
                for p, row in zip(chunk, z): out[p] = row.clone()
        assert max(float((out[p][0] - cache[p]).abs().max()) for p in paths[:40]) < 1e-5   # view 0 is the cached feature
        torch.save(out, REP_DIR / "simclr_views.pt"); print("saved", len(out), "images x", len(VIEWS), "views")


if __name__ == "__main__" and sys.argv[1:] == ["views"]: build_views()

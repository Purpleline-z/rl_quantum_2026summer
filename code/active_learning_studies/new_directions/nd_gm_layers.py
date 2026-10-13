"""Intermediate SimCLR layers (graph-mechanism study): global-average-pooled outputs of layer2 (128-d), layer3 (256-d) and layer4 (512-d, equals the final feature) of the frozen ResNet-18 for every
trajectory and ideal image.  Used ONLY to build alternative graph topologies; the head still reads the final 512-d features.  Label-free, CPU.   python3 nd_gm_layers.py build"""
from __future__ import annotations

import sys
import tempfile

import torch
from PIL import Image

import nd_core as core

LAYER_FILE = core.HERE / "results" / "feature_layers_all.pt"


def build() -> None:
    cache = core.feature_cache(); torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(8000, "A", scratch, cache); enc = ctx.features.base_model.encoder.eval(); tf = ctx.features.tf; kids = list(enc.children())
        paths = sorted(cache); out = {"l2": {}, "l3": {}, "l4": {}}
        with torch.no_grad():
            for i in range(0, len(paths), 16):
                chunk = paths[i:i + 16]; x = torch.stack([tf(Image.open(p).convert("L")) for p in chunk]); feats = {}
                for idx, m in enumerate(kids):
                    x = m(x)
                    if idx in (5, 6, 7): feats[{5: "l2", 6: "l3", 7: "l4"}[idx]] = x.mean(dim=(2, 3))
                for name, v in feats.items():
                    for p, row in zip(chunk, v): out[name][p] = row.clone()
        diff = max(float((out["l4"][p] - cache[p]).abs().max()) for p in paths[:50]); print("layer4 pooled vs cached final feature: max abs diff", diff)
        torch.save(out, LAYER_FILE); print("saved", len(paths), "images ->", LAYER_FILE)


def load() -> dict:
    return torch.load(LAYER_FILE, weights_only=False)


if __name__ == "__main__" and sys.argv[1:] == ["build"]: build()

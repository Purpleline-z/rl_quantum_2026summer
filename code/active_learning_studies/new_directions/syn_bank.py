"""Synthetic image bank: render with rheed_sim, encode with the project's frozen SimCLR encoder (same preprocessing as real images: 8-bit L -> Resize(224) -> Normalize), cache features.

  python3 syn_bank.py --tag v0 --n 250 --seed 1 [--styles traj] [--types 1x1,c6x2,RT13,HTR] [--threads 4]
Writes results/syn/bank_<tag>.pt = {feat (N,512), typ (N,), style (N,), seed, params (list of dicts)}; the images themselves are not stored (re-render with the same seed)."""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
import nd_core as core, rheed_sim as S

HERE = Path(__file__).resolve().parent; OUT = HERE / "results" / "syn"


def encoder_and_tf():
    cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(11100, "A", scratch, cache)
    return ctx.features.base_model.encoder.eval(), ctx.features.tf, cache


def encode_images(enc, tf, imgs):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 32): out.append(enc(torch.stack([tf(Image.fromarray(a, "L")) for a in imgs[i:i + 32]])))
    return torch.cat(out)


def build(tag, n, seed, styles=("traj",), types=S.TYPES, sim=S, store_images: int = 0):
    enc, tf, _ = encoder_and_tf(); feats, typ, sty, params, thumbs = [], [], [], [], []
    for style in styles:
        for t_i, t in enumerate(types):
            seed_t = seed * 1000 + t_i * 10 + (0 if style == "traj" else 1); gen = sim.generate(t, n, seed_t, style); imgs = [a for a, _ in gen]
            feats.append(encode_images(enc, tf, imgs)); typ += [t] * len(imgs); sty += [style] * len(imgs); params += [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()} for _, p in gen]
            thumbs += [np.array(Image.fromarray(a).resize((164, 123))) for a in imgs[:store_images]]
    OUT.mkdir(parents=True, exist_ok=True); torch.save({"feat": torch.cat(feats), "typ": typ, "style": sty, "seed": seed, "params": params}, OUT / f"bank_{tag}.pt"); return OUT / f"bank_{tag}.pt", thumbs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=250); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--styles", default="traj"); ap.add_argument("--types", default="1x1,c6x2,RT13,HTR"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); t0 = time.time(); path, _ = build(a.tag, a.n, a.seed, tuple(a.styles.split(",")), tuple(a.types.split(","))); print(path, f"{time.time() - t0:.0f}s")

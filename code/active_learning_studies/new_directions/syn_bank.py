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
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=250); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--styles", default="traj"); ap.add_argument("--types", default="1x1,c6x2,RT13,HTR"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--tune", default="global", help="global | types (per-type multipliers of results/syn/calibration_type_<T>.json)")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.tune == "types":
        for t in S.TYPES: S.TUNE_T[t] = json.loads((OUT / f"calibration_type_{t}.json").read_text())["best"]
    t0 = time.time(); path, _ = build(a.tag, a.n, a.seed, tuple(a.styles.split(",")), tuple(a.types.split(","))); print(path, f"{time.time() - t0:.0f}s")


def build_augreal(tag, m=25, seed=1, mask=True, threads=4):
    """Control bank without a simulator: ``m`` pixel-augmented copies (shift, scale, exposure, gamma, noise, blur) of every REAL ideal image (direct-beam masked), labelled with the image's own type.
    Same bank format as ``build`` (types '1x1', 'c6x2', 'RT13', 'HTR'; Twinned(2x1) ideals are skipped)."""
    import syn_ideal_fix
    from scipy.ndimage import gaussian_filter, shift as nd_shift, zoom
    torch.set_num_threads(threads); enc, tf, cache = encoder_and_tf(); paths = sorted(p for p in cache if "STO_ideal" in p); rng = np.random.default_rng(seed); imgs, typ = [], []
    name = {"STO_ideal_1x1": "1x1", "STO_ideal_c6x2": "c6x2", "STO_ideal_RT13": "RT13", "STO_ideal_HTR": "HTR"}
    for p in paths:
        key = p.split("/")[-2]
        if key not in name: continue
        base = syn_ideal_fix.load_fixed(p, mask).astype(np.float32)
        for _ in range(m):
            a = zoom(base, rng.uniform(0.95, 1.05), order=1); H, W = base.shape; out = np.zeros_like(base); h, w = min(H, a.shape[0]), min(W, a.shape[1]); out[:h, :w] = a[:h, :w]
            a = nd_shift(out, (rng.uniform(-15, 15), rng.uniform(-25, 25)), order=1, mode="constant"); a = np.clip(a / 255.0, 0, 1) ** rng.uniform(0.8, 1.2) * 255.0 * np.exp(rng.normal(0, 0.3))
            a = gaussian_filter(a, rng.uniform(0, 1.5)) + rng.normal(0, rng.uniform(0, 3), a.shape); a[: int(round(135 / 492 * H))] *= 0 if mask else 1; imgs.append(np.clip(a, 0, 255).astype(np.uint8)); typ.append(name[key])
    feats = encode_images(enc, tf, imgs); OUT.mkdir(parents=True, exist_ok=True); torch.save({"feat": feats, "typ": typ, "style": ["traj"] * len(typ), "seed": seed, "params": [{}] * len(typ)}, OUT / f"bank_{tag}.pt"); return OUT / f"bank_{tag}.pt"


def align_bank(src_tag: str, dst_tag: str, mode: str):
    """Feature-level domain alignment of a synthetic bank (raw 512-d SimCLR features):
      mean      shift the global mean of the bank to the mean of the real unlabelled trajectory frames
      coral     mean shift + CORAL (whiten with the bank's covariance, re-colour with the real unlabelled frames' covariance; both shrunk by 1e-2 * trace/d)
      typemean  per type: shift the type's mean to the mean of the real clean masked ideal images of that type (uses the real anchors only)
    Real references never include a labelled pair-group image (``syn_calibrate.references``)."""
    from syn_calibrate import references
    import nd_core as core
    enc, tf, mu, U, ideal, nu = references(); cache = core.feature_cache(); bank = torch.load(OUT / f"bank_{src_tag}.pt", weights_only=False); X = bank["feat"].float().clone(); typ = np.array(bank["typ"])
    import tempfile
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(11100, "A", scratch, cache)
    labelled_imgs = {str(x) for g in ctx.exp.groups.values() for r in g.itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    R = torch.stack([cache[p].float() for p in sorted(cache) if "Trajectories" in p and p not in labelled_imgs])
    if mode == "mean": X = X - X.mean(0) + R.mean(0)
    elif mode == "coral":
        def cov(A): A = A - A.mean(0); C = A.T @ A / (len(A) - 1); return C + 1e-2 * torch.trace(C) / C.shape[0] * torch.eye(C.shape[0])
        Cs, Cr = cov(X), cov(R); Ls, Lr = torch.linalg.cholesky(Cs), torch.linalg.cholesky(Cr); Z = (X - X.mean(0)) @ torch.linalg.inv(Ls).T; X = Z @ Lr.T + R.mean(0)
    elif mode == "typemean":
        import syn_ideal_fix
        fx = syn_ideal_fix.load(); idx = {p: i for i, p in enumerate(fx["paths"])}; from PIL import Image
        for t, key in zip(S.TYPES, ("1x1", "c6x2", "RT13", "HTR")):
            ps = [p for p in fx["paths"] if f"STO_ideal_{key}/" in p and not Image.open(p).mode.startswith("I;16")]; target = fx["mask"][[idx[p] for p in ps]].float().mean(0); m = typ == t; X[m] = X[m] - X[m].mean(0) + target
    else: raise ValueError(mode)
    bank = dict(bank); bank["feat"] = X; bank["aligned"] = mode; torch.save(bank, OUT / f"bank_{dst_tag}.pt"); return OUT / f"bank_{dst_tag}.pt"


def build_tmgauss(src_tag="v1_typemean", dst_tag="v1_tmgauss", n=400, seed=5):
    """Control bank: Gaussian features with the per-type mean and covariance of an (aligned) synthetic bank (the simulator's particular images, i.e. everything beyond mean and covariance, are discarded)."""
    bank = torch.load(OUT / f"bank_{src_tag}.pt", weights_only=False); X = bank["feat"].float(); typ = np.array(bank["typ"]); g = torch.Generator().manual_seed(seed); feats, tt = [], []
    for t in S.TYPES:
        A = X[typ == t]; mu = A.mean(0); C = (A - mu).T @ (A - mu) / (len(A) - 1); lam, V = torch.linalg.eigh(C.double()); Lh = (V * lam.clamp_min(0).sqrt()).float(); z = torch.randn(n, A.shape[1], generator=g); feats.append(mu + z @ Lh.T); tt += [t] * n
    torch.save({"feat": torch.cat(feats), "typ": tt, "style": ["traj"] * len(tt), "seed": seed, "params": [{}] * len(tt), "aligned": "gauss"}, OUT / f"bank_{dst_tag}.pt"); return OUT / f"bank_{dst_tag}.pt"

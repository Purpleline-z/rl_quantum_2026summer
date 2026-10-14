"""Calibrate the simulator's global multipliers (rheed_sim.TUNE) so that synthetic images fall where real ones do in the frozen-SimCLR feature space.

Real references (identity rule: NO image that is part of any labelled pair group, so no frame that can be held out in any seed): the 840 unlabelled trajectory frames, and the clean real ideal images
(direct-beam-masked variant of syn_ideal_fix, the trajectory-frame style; 16-bit PNGs excluded).  Loss = precision (median distance synthetic -> nearest real reference) + recall (median distance real unlabelled
frame -> nearest synthetic) + typed recall (mean over types of the median distance real ideal image of that type -> nearest synthetic image OF THE SAME TYPE).  Coordinate search over multiplicative steps.
  python3 syn_calibrate.py --n 40 --sweeps 2 --out results/syn/calibration.json"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
import nd_core as core, rheed_sim as S, syn_ideal_fix
from syn_bank import encoder_and_tf, encode_images, OUT

TYPES = S.TYPES


def references():
    enc, tf, cache = encoder_and_tf(); paths = sorted(cache); X = torch.stack([cache[p].float() for p in paths]); mu = X.mean(0)
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(11100, "A", scratch, cache)
    labelled_imgs = {str(x) for g in ctx.exp.groups.values() for r in g.itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    unl = [p for p in paths if "Trajectories" in p and p not in labelled_imgs]
    fx = syn_ideal_fix.load(); idx = {p: i for i, p in enumerate(fx["paths"])}; nrm = lambda x: (x - mu) / (x - mu).norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ideal = {}
    for t, key in zip(TYPES, ("1x1", "c6x2", "RT13", "HTR")):
        ps = [p for p in fx["paths"] if f"STO_ideal_{key}/" in p and not Image.open(p).mode.startswith("I;16")]; ideal[t] = nrm(fx["mask"][[idx[p] for p in ps]].float())
    return enc, tf, mu, nrm(torch.stack([cache[p].float() for p in unl])), ideal, len(unl)


def loss_of(enc, tf, mu, U, ideal, n, seed=7, tune=None, return_parts=False):
    old = dict(S.TUNE)
    if tune: S.TUNE.update(tune)
    try:
        feats, typ = [], []
        for i, t in enumerate(TYPES):
            gen = S.generate(t, n, seed * 100 + i, "traj"); feats.append(encode_images(enc, tf, [a for a, _ in gen])); typ += [t] * n
    finally: S.TUNE.clear(); S.TUNE.update(old)
    F = torch.cat(feats); F = (F - mu) / (F - mu).norm(dim=-1, keepdim=True).clamp_min(1e-9); typ = np.array(typ)
    real_all = torch.cat([U] + list(ideal.values())); prec = float((1 - (F @ real_all.T).max(1).values).median()); rec = float((1 - (U @ F.T).max(1).values).median())
    typed = float(np.mean([float((1 - (ideal[t] @ F[typ == t].T).max(1).values).median()) for t in TYPES]))
    parts = {"precision": prec, "recall": rec, "typed_recall": typed}
    return (prec + rec + typed, parts) if return_parts else prec + rec + typed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--sweeps", type=int, default=2); ap.add_argument("--out", default=str(OUT / "calibration.json")); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--init", default=None); a = ap.parse_args()
    torch.set_num_threads(a.threads); enc, tf, mu, U, ideal, nu = references(); print("real unlabelled frames", nu, {t: len(v) for t, v in ideal.items()}, flush=True)
    tune = dict(S.TUNE)
    if a.init: tune.update(json.loads(Path(a.init).read_text())["best"])
    best, parts = loss_of(enc, tf, mu, U, ideal, a.n, tune=tune, return_parts=True); hist = [{"tune": dict(tune), "loss": best, **parts}]; print("start", round(best, 4), parts, flush=True)
    for sweep in range(a.sweeps):
        for key in list(tune):
            for f in (0.5, 2.0) if sweep % 2 == 0 else (0.7, 1.4):
                cand = dict(tune); cand[key] = tune[key] * f; l, parts = loss_of(enc, tf, mu, U, ideal, a.n, tune=cand, return_parts=True); hist.append({"tune": cand, "loss": l, **parts})
                if l < best - 1e-4: best, tune = l, cand; print(f"sweep {sweep} {key} x{f}: loss {l:.4f} {parts}", flush=True)
            Path(a.out).write_text(json.dumps({"best": tune, "best_loss": best, "history": hist}, indent=1))
    print("final", tune, best)

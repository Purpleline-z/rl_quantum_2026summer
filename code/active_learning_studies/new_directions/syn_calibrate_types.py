"""Per-type refinement of the simulator calibration (rheed_sim.TUNE_T): for one type, coordinate search of multiplicative factors so that the synthetic images of that type match the REAL ideal images of that type
(direct-beam-masked, clean 8-bit ones; these are anchors of every learner, so no held-out pair-group information is used).  Loss = median distance real ideal -> nearest synthetic of the type (recall)
+ median distance synthetic -> nearest real ideal of the type (precision) + 0.5 * median distance synthetic -> nearest real unlabelled trajectory frame.
  python3 syn_calibrate_types.py --type HTR --n 40 --sweeps 2 --out results/syn/calibration_type_HTR.json"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import rheed_sim as S
from syn_bank import encode_images, OUT
from syn_calibrate import references

KEYS = ["sy", "sx", "sup", "spec", "gain", "bg", "blur", "kik", "kikw", "noise", "rodw"]


def loss_type(enc, tf, mu, U, ideal, typ, n, mult, seed=21):
    S.TUNE_T[typ] = dict(mult)
    gen = S.generate(typ, n, seed, "traj"); F = encode_images(enc, tf, [a for a, _ in gen]); F = (F - mu) / (F - mu).norm(dim=-1, keepdim=True).clamp_min(1e-9)
    r = ideal[typ]; rec = float((1 - (r @ F.T).max(1).values).median()); prec = float((1 - (F @ r.T).max(1).values).median()); pu = float((1 - (F @ U.T).max(1).values).median())
    return rec + prec + 0.5 * pu, {"recall": rec, "precision_ideal": prec, "precision_unl": pu}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--type", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--sweeps", type=int, default=2); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--out", default=None); a = ap.parse_args()
    torch.set_num_threads(a.threads); enc, tf, mu, U, ideal, nu = references(); mult = {k: 1.0 for k in KEYS}; best, parts = loss_type(enc, tf, mu, U, ideal, a.type, a.n, mult); hist = [{"mult": dict(mult), "loss": best, **parts}]; print("start", round(best, 4), parts, flush=True)
    for sweep in range(a.sweeps):
        for key in KEYS:
            for f in (0.5, 2.0) if sweep % 2 == 0 else (0.7, 1.4):
                cand = dict(mult); cand[key] = mult[key] * f; l, parts = loss_type(enc, tf, mu, U, ideal, a.type, a.n, cand); hist.append({"mult": cand, "loss": l, **parts})
                if l < best - 1e-4: best, mult = l, cand; print(f"sweep {sweep} {key} x{f}: loss {l:.4f} {parts}", flush=True)
        Path(a.out or OUT / f"calibration_type_{a.type}.json").write_text(json.dumps({"type": a.type, "best": mult, "best_loss": best, "history": hist}, indent=1))
    print("final", a.type, mult, round(best, 4))

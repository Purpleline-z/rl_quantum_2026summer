"""Sim-to-real gap in the frozen-SimCLR feature space (the space the learners and the kNN graph use).

  python3 syn_gap.py --tag v0 [--md results/syn/gap_v0.md]
Features of real images are the cached features of the learners (centred with the mean of all cached real features, L2-normalised; distance = 1 - cos, as in the graph and rt_a1b).
Reported: (1) per type, median distance from synthetic images to the nearest real trajectory frame, to the nearest real ideal image of the same type and of another type, with the real-real references
(ideal -> nearest trajectory frame; ideal -> nearest other ideal of its type); (2) real-vs-synthetic logistic-regression AUC (5-fold CV); (3) accuracy of a type classifier trained on synthetic
features only when applied to the real ideal images (and the real-only leave-one-out reference); (4) the real ideal images of 16-bit PNG format (saturated by the loader) are reported separately and excluded
from the 'clean real' set."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
TYPES = ("1x1", "c6x2", "RT13", "HTR")

def norm(x, mu):
    x = x - mu; return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)

import nd_core as core
from syn_bank import OUT
from tg_md import md

TYPES = ("1x1", "c6x2", "RT13", "HTR")


def gap_report(bank, label="") -> str:
    """Real references = the 840 unlabelled trajectory frames (no labelled pair-group image) and the clean real ideal images in their direct-beam-masked (trajectory-frame style) variant."""
    from syn_calibrate import references
    enc, tf, mu, U, ideal, nu = references(); out = []; S = norm(bank["feat"].float(), mu); typ = np.array(bank["typ"]); rows = []
    allideal = torch.cat(list(ideal.values()))
    for t in TYPES:
        m = typ == t
        if not m.any(): continue
        s = S[m]; d_traj = 1 - (s @ U.T).max(1).values; d_same = 1 - (s @ ideal[t].T).max(1).values
        d_other = torch.stack([1 - (s @ ideal[o].T).max(1).values for o in TYPES if o != t], 1).min(1).values
        r = ideal[t]; d_rs = 1 - (r @ s.T).max(1).values; sim = r @ r.T; sim.fill_diagonal_(-2); d_ii = 1 - sim.max(1).values
        rows.append({"type": t, "n_syn": int(m.sum()), "syn->nearest unlabelled frame": float(d_traj.median()), "syn->nearest ideal, same type": float(d_same.median()), "syn->nearest ideal, other type": float(d_other.median()),
                     "REAL ideal->nearest syn (same type)": float(d_rs.median()), "REAL ideal->nearest other ideal (same type)": float(d_ii.median()), "REAL ideal->nearest unlabelled frame": float((1 - (r @ U.T).max(1).values).median()), "n_real_ideal": len(r)})
    out.append(f"#### Distances (1 - cos of centred features; medians) {label}\n"); out.append(md(pd.DataFrame(rows), "{:.3f}"))
    rec = float((1 - (U @ S.T).max(1).values).median()); prec = float((1 - (S @ torch.cat([U, allideal]).T).max(1).values).median())
    out.append(f"\nPrecision (synthetic -> nearest real reference, median) = {prec:.3f}; recall (real unlabelled frame -> nearest synthetic, median) = {rec:.3f}; for reference the median nearest-neighbour distance between real unlabelled frames is {float(np.median((1 - (U @ U.T).fill_diagonal_(-2).max(1).values).numpy())):.4f}.\n")
    real = torch.cat([U, allideal]).numpy(); syn = S.numpy(); rng = np.random.default_rng(0); idx = rng.choice(len(syn), size=min(len(syn), 2 * len(real)), replace=False)
    Xr = np.concatenate([real, syn[idx]]); yr = np.r_[np.zeros(len(real)), np.ones(len(idx))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)); p = cross_val_predict(clf, Xr, yr, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    out.append(f"**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, {len(real)} real [unlabelled frames + masked clean ideal images] vs {len(idx)} synthetic): AUC = {roc_auc_score(yr, p):.3f} (0.5 = indistinguishable).\n")
    Xi = allideal.numpy(); yi = np.concatenate([[t] * len(ideal[t]) for t in TYPES]); c2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(syn, typ); pred = c2.predict(Xi)
    cm = pd.crosstab(pd.Series(yi, name="real ideal (true)"), pd.Series(pred, name="predicted by synthetic-trained classifier")); c3 = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)); loo = cross_val_predict(c3, Xi, yi, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out.append(f"**Type classifier trained on synthetic features only, applied to the {len(yi)} clean real ideal images (masked):** accuracy = {np.mean(pred == yi):.3f} (majority-class rate {max(np.bincount(pd.factorize(yi)[0])) / len(yi):.3f}); real-only 5-fold reference = {np.mean(loo == yi):.3f}.\n")
    cm = cm.reset_index(); cm.columns = [str(c) for c in cm.columns]; out.append(md(cm.astype({c: float for c in cm.columns[1:]}), "{:.0f}")); return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--md", default=None); a = ap.parse_args()
    bank = torch.load(OUT / f"bank_{a.tag}.pt", weights_only=False); text = gap_report(bank, f"(bank `{a.tag}`)"); print(text)
    if a.md: Path(a.md).write_text(text)

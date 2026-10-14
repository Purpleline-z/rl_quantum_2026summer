"""Zero-shot anchor matching (descriptive, development, no pair labels used anywhere): for every decisive labelled pair judgment (type k) predict the winner by the sign of log q_k(img1) - log q_k(img2), q = class posterior of the image
(logistic regression on all 144 typed ideal images; features z-scored over all images) in physics / SimCLR / random-projection feature space.  Reported overall, by judgment type and by anchor distance (SimCLR geometry, far = both images > 0.01
from every ideal image).   python3 pf_zeroshot.py [--out FILE.md]"""
import argparse, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import nd_core as core, nd_physics as NP, nd_graph_learner as G, rt_core as rt

warnings.filterwarnings("ignore")


def main(out):
    cache = core.feature_cache(); S = NP._load(); paths = S["paths"]; idx = S["index"]; cols = list(G.ACTIVE)
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(10600, "A", scratch, cache)
    e = ctx.exp; ideal = [(str(p), core.frozen.TYPE_TO_INDEX[c]) for tab in (e.references, e.utility_images, e.test_images) for c, ps in tab.items() for p in ps if core.frozen.TYPE_TO_INDEX[c] in cols]
    rows = e.rows_for(list(e.groups)); rows = rows[rows.Winner.isin(["1", "2"])].reset_index(drop=True); sign = np.where((rows.Winner == "1").to_numpy(), 1.0, -1.0); k = rows.type_idx.to_numpy().astype(int); w = rows.confidence_weight.to_numpy()
    tables = {"SimCLR (final layer)": np.stack([cache[p].float().numpy() for p in paths]), **{f"physics {n}": S["raw"][n] for n in ("pfa2", "pfb2", "pfc2")}}
    tables["random projection of SimCLR (dim of pfa)"] = tables["SimCLR (final layer)"] @ S["proj"].numpy()
    sim = rt.Sim(cache); allanc = [p for p, _ in ideal]; a1 = [str(x) for x in rows.resolved_img1]; a2 = [str(x) for x in rows.resolved_img2]; amin = np.minimum(1 - sim.max_sim(a1, allanc), 1 - sim.max_sim(a2, allanc)); far = amin > 0.01
    tr = [i for i, p in enumerate(paths) if "/Trajectories/" in p]; res = []
    for name, X in tables.items():
        Z = np.clip((X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6), -5, 5); ya = np.array([c for _, c in ideal]); clf = LogisticRegression(C=1.0, max_iter=3000).fit(Z[[idx[p] for p, _ in ideal]], ya)
        def post(ps):
            pr = clf.predict_proba(Z[[idx[p] for p in ps]]); out_ = np.full((len(ps), len(cols)), 1e-6)
            for j, c in enumerate(clf.classes_): out_[:, cols.index(int(c))] = pr[:, j]
            return out_ / out_.sum(1, keepdims=True)
        p1, p2 = post(a1), post(a2); col = np.array([cols.index(int(x)) if int(x) in cols else 0 for x in k]); r = np.arange(len(col)); d = np.log(p1[r, col] + 1e-4) - np.log(p2[r, col] + 1e-4)
        for label, m in (("all", np.ones(len(d), bool)), ("anchor-near", ~far), ("anchor-far", far), ("(1x1)", k == 0), ("c(6x2)", k == 2), ("RT13", k == 3), ("HTR", k == 4)):
            auc = roc_auc_score(sign[m] > 0, d[m]) if len(set(sign[m])) == 2 else np.nan; res.append({"features": name, "subset": label, "n": int(m.sum()), "accuracy": float(((d[m] * sign[m]) > 0).mean()), "AUC": float(auc)})
    t = pd.DataFrame(res); wide = t.pivot(index="subset", columns="features", values=["accuracy", "AUC"]); n = t[t.features == t.features.iloc[0]].set_index("subset").n
    lines = ["| subset | n | " + " | ".join(f"{m} {f}" for m in ("accuracy", "AUC") for f in tables) + " |", "|" + "---|" * (2 + 2 * len(tables))]
    for s in ["all", "anchor-near", "anchor-far", "(1x1)", "c(6x2)", "RT13", "HTR"]: lines.append(f"| {s} | {n[s]} | " + " | ".join(f"{wide[(m, f)][s]:.3f}" for m in ("accuracy", "AUC") for f in tables) + " |")
    print("\n".join(lines)); (Path(out).write_text("\n".join(lines) + "\n") if out else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); main(ap.parse_args().out)

"""Data-level diagnostic (design-informing; uses labels of all 521 rows, so it is NOT evidence about acquisition rules):
does adding the 824 unlabelled trajectory images (never in any pair) to the image graph improve prediction of the judgment outcome (decisive / tie / not_apply)
from the 8 label-propagated type-posterior features of graph_typed.ImageGraph?  Grouped 5-fold CV by pair group, repeated; references = random 60% of ideal images per class.
usage: python3 diagnostic_outcomes.py [repeats]"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[6] if (Path(__file__).resolve().parents[6] / "data").exists() else Path(__file__).resolve().parents[5]
REP = ROOT / "code/active_learning_studies/image_representation_analysis/results/representation_exploration"
man = pd.read_csv(REP / "manifest.csv"); man["key"] = man.path.str.replace(r"^.*/(original data/.*)$", r"\1", regex=True)
X = np.load(REP / "cache/rheed_simclr_resnet18.npy").astype(np.float64); idx = {k: i for i, k in enumerate(man.key)}
LAB = {"(1 x 1)": 0, "c(6 x 2)": 1, "RT13": 2, "HTR": 3}; CSV = {"(1 x 1)": 0, "c(6 x 2)": 1, "(√13 x √13)": 2, "HTR": 3}
d = pd.read_csv(ROOT / "data/original data/Quantum Label Data - Pairwise_Comparisonv1.8.csv"); d = d[d.Reconstruction_Type.isin(CSV)].copy()
d["key1"] = "original data/" + d.Image1_Path; d["key2"] = "original data/" + d.Image2_Path
d = d[d.key1.isin(idx) & d.key2.isin(idx)].copy(); d["a"] = d.key1.map(idx); d["b"] = d.key2.map(idx); d["t"] = d.Reconstruction_Type.map(CSV); d["grp"] = d.key1 + "|" + d.key2
print("rows", len(d), "groups", d.grp.nunique(), "outcomes", d.Winner.value_counts().to_dict())
pair_nodes = sorted(set(d.a) | set(d.b)); ideal = man.index[(man.source == "ideal") & man.label.isin(LAB)].tolist(); traj_all = man.index[man.source == "trajectory"].tolist()
unl = [i for i in traj_all if i not in set(pair_nodes)]
Z = X - X.mean(0); Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True); S = Zn @ Zn.T; np.fill_diagonal(S, -9)
dup = {i for i in unl if S[i, pair_nodes + ideal].max() > 0.9999}
unl = [i for i in unl if i not in dup]; print("pair images", len(pair_nodes), "ideal", len(ideal), "unlabelled trajectory", len(unl), "dropped near-duplicates", len(dup))


def graph(nodes, k, alpha, refs):
    nodes = np.array(nodes); n = len(nodes); s = S[np.ix_(nodes, nodes)]; nb = np.argsort(-s, 1)[:, :k]
    a = np.zeros((n, n)); a[np.repeat(np.arange(n), k), nb.ravel()] = 1; a = np.maximum(a, a.T)
    dg = np.maximum(a.sum(1), 1); sn = a / np.sqrt(np.outer(dg, dg)); pos = {int(v): i for i, v in enumerate(nodes)}
    y = np.zeros((n, 4)); 
    for r, t in refs.items(): y[pos[r], t] = 1
    f = np.maximum(np.linalg.solve(np.eye(n) - alpha * sn, y), 0); q = f / np.maximum(f.sum(1, keepdims=True), 1e-9); return q, pos


def feats(q, pos):
    out = []
    for a, b, t in zip(d.a, d.b, d.t):
        qa, qb = q[pos[a]], q[pos[b]]; out.append([qa[t], qb[t], abs(qa[t] - qb[t]), qa[t] + qb[t], min(qa[t], qb[t]), max(qa[t], qb[t]), np.abs(qa - qb).sum(), float(qa @ qb)])
    return np.array(out)


def cv(F, y, folds, C=1.0):
    P = np.zeros(len(y)); onehot = np.eye(4)[d.t.values]
    for f in range(5):
        tr, te = folds != f, folds == f; mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        m = LogisticRegression(C=C, max_iter=2000).fit(np.column_stack([onehot[tr], (F[tr] - mu) / sd]), y[tr]); P[te] = m.predict_proba(np.column_stack([onehot[te], (F[te] - mu) / sd]))[:, 1]
    return roc_auc_score(y, P)


def main(reps=10):
    ys = {o: (d.Winner.isin(["1", "2"]) if o == "decisive" else d.Winner == o).values.astype(int) for o in ("decisive", "tie", "not_apply")}
    B, U = pair_nodes + ideal, pair_nodes + ideal + unl
    variants = [("base k10 a.9", B, 10, .9), ("base k10 a.99", B, 10, .99), ("base k20 a.99", B, 20, .99), ("base k20 a.999", B, 20, .999), ("base k40 a.99", B, 40, .99),
                ("+unl k10 a.9", U, 10, .9), ("+unl k10 a.99", U, 10, .99), ("+unl k20 a.99", U, 20, .99), ("+unl k20 a.999", U, 20, .999), ("+unl k40 a.99", U, 40, .99), ("+unl k40 a.999", U, 40, .999)]
    res = {v[0]: {o: [] for o in ys} for v in variants}; groups = d.grp.unique()
    for r in range(reps):
        rng = np.random.default_rng(r); refs = {}
        for c in LAB.values():
            ids = [i for i in ideal if LAB[man.label[i]] == c]; ids = list(rng.permutation(ids)); refs.update({int(i): c for i in ids[:int(round(.6 * len(ids)))]})
        folds = d.grp.map(dict(zip(groups, rng.permutation(len(groups)) % 5))).values
        for name, nodes, k, alpha in variants:
            nodes = [n for n in nodes]; q, pos = graph(nodes, k, alpha, refs); F = feats(q, pos)
            for o, y in ys.items(): res[name][o].append(cv(F, y, folds))
    print(f"\nAUC (mean over {reps} repeats; sd) of outcome prediction from 8 propagated type-posterior features + type one-hot")
    print(f"{'graph':24s}" + "".join(f"{o:>16s}" for o in ys))
    for name in res: print(f"{name:24s}" + "".join(f"{np.mean(res[name][o]):>10.3f}({np.std(res[name][o]):.3f})" for o in ys))


if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)

"""Validation of the physics features on the ideal images: do they separate the four types (cross-validated; the seed's 28 outer-test ideal images are left out, as in the paper's classification endpoint) better than SimCLR 1-NN?
  python3 pf_ideal_cv.py [--seeds 10600-10619] [--blocks name,...]   -> prints accuracy tables (generated; nothing typed by hand)"""
import argparse, tempfile, warnings
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
import nd_core as core

warnings.filterwarnings("ignore")
CLASSES = ["(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR"]


def splits(seeds):
    cache = core.feature_cache(); out = {}
    with tempfile.TemporaryDirectory() as s:
        for seed in seeds:
            e = core.make_ctx(seed, "A", s, cache).exp
            out[seed] = {"train": [(str(p), c) for table in (e.references, e.utility_images) for c, ps in table.items() for p in ps if c in CLASSES], "test": [(str(p), c) for c, ps in e.test_images.items() for p in ps if c in CLASSES]}
    return out


def zscore(X, fit_rows):
    mu, sd = X[fit_rows].mean(0), X[fit_rows].std(0) + 1e-6; return (X - mu) / sd


def evaluate(spl, tables: dict, index: dict, traj_rows, is16) -> pd.DataFrame:
    """tables: name -> (n_images, d) matrix aligned to ``index``.  Reports 1-NN (cosine on centred z-scored features) and logistic regression accuracy of the seed's outer-test ideals."""
    rows = []
    for name, X in tables.items():
        for seed, sp in spl.items():
            tr = [index[p] for p, _ in sp["train"]]; te = [index[p] for p, _ in sp["test"]]; ytr = np.array([CLASSES.index(c) for _, c in sp["train"]]); yte = np.array([CLASSES.index(c) for _, c in sp["test"]])
            Z = zscore(X, np.concatenate([tr, traj_rows])); Zc = Z - Z[tr].mean(0); Zn = Zc / np.linalg.norm(Zc, axis=1, keepdims=True).clip(1e-9)
            nn = ytr[(Zn[te] @ Zn[tr].T).argmax(1)]
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], ytr).predict(Z[te]); t16 = np.array([is16[i] for i in te])
            for clf, pred in (("1nn", nn), ("logreg", lr)):
                r = {"block": name, "clf": clf, "seed": seed, "acc": float((pred == yte).mean()), "acc_8bit": float((pred == yte)[~t16].mean()) if (~t16).any() else np.nan}
                for k, c in enumerate(CLASSES): r[f"acc_{c}"] = float((pred == yte)[yte == k].mean()) if (yte == k).any() else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def load_all():
    b = torch.load(core.HERE / "results" / "pf" / "blocks_v1.pt", weights_only=False); paths = b["paths"]; index = {p: i for i, p in enumerate(paths)}
    traj_rows = np.array([i for i, p in enumerate(paths) if "/Trajectories/" in p]); ex = torch.load(core.HERE / "results" / "pf" / "extract.pt", weights_only=False); is16 = [ex[p]["mode"] == "I;16" for p in paths]
    return b["blocks"], paths, index, traj_rows, is16


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="10600-10619"); ap.add_argument("--sets", default=""); args = ap.parse_args(); lo, hi = args.seeds.split("-"); spl = splits(range(int(lo), int(hi) + 1))
    blocks, paths, index, traj_rows, is16 = load_all(); cache = core.feature_cache(); S4 = torch.stack([cache[p].float() for p in paths]).numpy()
    sets = {"simclr": S4, **{k: v for k, v in blocks.items() if k != "spacing"}, "pxprof+bg": np.hstack([blocks["pxprof"], blocks["bg"]]), "uprof+peaks": np.hstack([blocks["uprof"], blocks["peaks"]]), "all_pf": np.hstack([blocks[k] for k in blocks if k != "spacing"] + [blocks["spacing"]])}
    df = evaluate(spl, sets, index, traj_rows, is16); t = df.groupby(["block", "clf"]).mean(numeric_only=True).drop(columns="seed").reset_index(); print(t.round(3).to_string(index=False))

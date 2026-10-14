"""Validation of the physics features on the ideal images: do they separate the four types (cross-validated; the seed's 28 outer-test ideal images are left out, as in the paper's classification endpoint) better than SimCLR 1-NN?
  python3 pf_ideal_cv.py [--seeds 10600-10619] [--blocks name,...]   -> prints accuracy tables (generated; nothing typed by hand)"""
import argparse, tempfile, warnings
from pathlib import Path
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


PER: list = []


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
                for q, ok in zip(te, (pred == yte)): PER.append({"block": name, "clf": clf, "image": q, "ok": float(ok), "seed": seed})
                r = {"block": name, "clf": clf, "seed": seed, "acc": float((pred == yte).mean()), "acc_8bit": float((pred == yte)[~t16].mean()) if (~t16).any() else np.nan}
                for k, c in enumerate(CLASSES): r[f"acc_{c}"] = float((pred == yte)[yte == k].mean()) if (yte == k).any() else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def load_all():
    b = torch.load(core.HERE / "results" / "pf" / "blocks_v2.pt", weights_only=False); paths = b["paths"]; index = {p: i for i, p in enumerate(paths)}
    traj_rows = np.array([i for i, p in enumerate(paths) if "/Trajectories/" in p]); ex = torch.load(core.HERE / "results" / "pf" / "extract.pt", weights_only=False); is16 = [ex[p]["mode"] == "I;16" for p in paths]
    return b["blocks"], paths, index, traj_rows, is16


def build_sets(version: str):
    import nd_physics as NP
    b = torch.load(core.HERE / "results" / "pf" / f"blocks_{version}.pt", weights_only=False); paths = b["paths"]; bl = b["blocks"]; cache = core.feature_cache(); S4 = torch.stack([cache[p].float() for p in paths]).numpy()
    F16 = torch.stack([NP._load()["f16"][p] for p in paths]).numpy(); pfa = np.hstack([bl["pxprof"], bl["bg"], bl["width"]]); pfb = np.hstack([bl["peaks"], bl["fourier"], bl["bg"], bl["width"], bl["spacing"]]); pfc = np.hstack([bl[k] for k in ("pxprof", "uprof", "peaks", "fourier", "bg", "width", "spacing")])
    rng = np.random.default_rng(0); P_ = S4 @ (rng.standard_normal((512, pfa.shape[1])) / 512 ** .5)
    return paths, {"SimCLR final feature (as used by all learners)": S4, "SimCLR, 16-bit ideals decoded correctly": F16, "physics pfa": pfa, "physics pfb (peak table)": pfb, "physics pfc (all blocks)": pfc, "random projection of SimCLR (dim of pfa)": P_, "SimCLR + pfa (z-scored concat)": np.hstack([S4, pfa]), "SimCLR (16-bit decoded) + pfa": np.hstack([F16, pfa])}


def paired_images(base="SimCLR final feature (as used by all learners)"):
    """Per ideal image: mean correctness over the seeds in which it is an outer-test image; paired comparison of every feature set with ``base`` over images (Wilcoxon, bootstrap interval)."""
    from scipy.stats import wilcoxon
    df = pd.DataFrame(PER); m = df.groupby(["block", "clf", "image"]).ok.mean().unstack(0); rows = []; rng = np.random.default_rng(0)
    for clf in ("1nn", "logreg"):
        b = m.loc[clf][base]
        for name in m.loc[clf].columns:
            if name == base: continue
            d = (m.loc[clf][name] - b).dropna().to_numpy(); bs = rng.choice(d, (4000, len(d))).mean(1)
            try: p = float(wilcoxon(d).pvalue)
            except ValueError: p = 1.0
            rows.append({"clf": clf, "feature set": name, "mean diff vs SimCLR (accuracy per image)": float(d.mean()), "95% CI": f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]", "Wilcoxon p (uncorrected)": p, "n images": len(d)})
    t = pd.DataFrame(rows); cols = list(t.columns); return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f"{v:+.3f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in t.itertuples(index=False)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="10600-10619"); ap.add_argument("--version", default="v2"); ap.add_argument("--out", default=None); args = ap.parse_args(); lo, hi = args.seeds.split("-"); spl = splits(range(int(lo), int(hi) + 1))
    _, _, index, traj_rows, is16 = load_all(); paths, sets = build_sets(args.version); assert paths == sorted(index, key=index.get)
    df = evaluate(spl, sets, index, traj_rows, is16); t = df.groupby(["block", "clf"]).agg(["mean"]).droplevel(1, axis=1).drop(columns="seed").reset_index(); t["n_seeds"] = len(spl)
    cols = ["block", "clf", "acc", "acc_8bit", "acc_(1 x 1)", "acc_c(6 x 2)", "acc_(√13 x √13)", "acc_HTR", "n_seeds"]; t = t[cols]; order = list(sets); t = t.sort_values(["clf", "block"], key=lambda c: c.map({k: i for i, k in enumerate(order)}) if c.name == "block" else c)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    print("\n".join(md)); (Path(args.out).write_text("\n".join(md) + "\n") if args.out else None); pm = paired_images(); print(); print(pm); (Path(args.out).with_name(Path(args.out).stem + "_paired.md").write_text(pm + "\n") if args.out else None)

"""Tables of the graph-GP study from the stage-1 cells (nothing typed by hand).

  python3 gp_tables.py dev DIR [DIR ...] [--ref=lap3_k10_st4] [--only=a,b]    levels and paired gains of every learner variant over a reference (+ the calibrated reference)
  python3 gp_tables.py timing DIR [DIR ...]                                   seconds per fit by learner and budget
  python3 gp_tables.py variance DIR [DIR ...] [--learner=gp_cv]               predictive sd by anchor-distance bin; does a variance-aware score help far from anchors?
  python3 gp_tables.py confirm SPEC.json DIR [DIR ...]                        pre-registered confirmation (Holm, success rule)
"""
import glob, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
import nd_calibration as C, cal_tables as T

warnings.filterwarnings("ignore")
CAL_CONSTANTS = {0: 1.352, 2: 0.682, 3: 0.293, 4: 0.630}      # per-type temperatures of the calibration study (CAL1 constants, fitted on calibration development seeds)
ANCHOR_BINS = (0.001, 0.01, 0.03)


def load_cells(dirs, seeds=None): return T.load_cells(dirs, seeds)


def apply_t(d, test, consts): return d / np.array([consts[int(k)] for k in test["k"]])


def variants_of(rec, gp_consts=None) -> dict:
    """name -> logit gap for every learner of one label set (GP learners: mean, probit-corrected, +calibration variants)."""
    out = {}; test = rec["test"]
    for name, r in rec["learners"].items():
        mu = np.asarray(r["mu"], float)
        if r["var"] is None:
            out[name] = mu; out[name + "+Tcal"] = apply_t(mu, test, CAL_CONSTANTS)
        else:
            var = np.asarray(r["var"], float); out[name + "|mean"] = mu; out[name + "|probit"] = mu / np.sqrt(1 + np.pi * var / 8)
            if gp_consts and name in gp_consts:
                out[name + "|probit+Tgp"] = apply_t(out[name + "|probit"], test, gp_consts[name]); out[name + "|mean+Tfix"] = apply_t(mu, test, gp_consts[name])      # fixed development constants
                if r.get("oof") is not None: out[name + "|mean+Tprior"] = apply_t(mu, test, C.fit_type_t(r["oof"], 8.0, gp_consts[name]))                       # within-run (cross-fitted) ridge towards the development constants
    return out


def strat_auc(d, test, mask):
    s = test["sign"][mask]
    return float(roc_auc_score(s > 0, d[mask])) if len(set(s)) == 2 and mask.sum() >= 4 else np.nan


def scores(d, test) -> dict:
    out = C.scores(d, test); far = test["g_anc_min"] > C.FAR_THRESHOLD; d = np.asarray(d, float)
    out["far_auc"] = strat_auc(d, test, far); out["near_auc"] = strat_auc(d, test, ~far)
    for lo, hi, tag in ((0, 0.001, "a1"), (0.001, 0.01, "a2"), (0.01, 0.03, "a3"), (0.03, 9, "a4")):
        m = (test["g_anc_min"] > lo) & (test["g_anc_min"] <= hi) if lo > 0 else (test["g_anc_min"] <= hi); out[f"{tag}_auc"] = strat_auc(d, test, m)
    return out


def frame(cells, gp_consts=None, draws=None) -> tuple[pd.DataFrame, dict]:
    rows, bins = [], {}
    for cell in cells:
        for (budget, draw), rec in cell["recs"].items():
            if draws is not None and draw not in draws: continue
            for name, d in variants_of(rec, gp_consts).items():
                sc = scores(d, rec["test"]); b = {k: sc.pop(k) for k in ("bin_w", "bin_conf", "bin_acc")}; rows.append({"seed": cell["seed"], "split": cell["split"], "budget": budget, "draw": draw, "variant": name, **sc})
                key = (cell["seed"], cell["split"], name); acc = bins.setdefault(key, [np.zeros(C.N_BINS)] * 3); bins[key] = [a + b[k] for a, k in zip(acc, ("bin_w", "bin_conf", "bin_acc"))]
    return pd.DataFrame(rows), bins


def per_seed(df, bins) -> pd.DataFrame:
    g = df.groupby(["seed", "split", "variant"]); cols = [c for c in df.columns if c.endswith(("_ll", "_acc", "_auc")) and not c.startswith(("far_", "near_")) or c in ("ll", "acc", "auc", "far_auc", "near_auc")]
    out = g[cols].mean()
    for tag in ("far", "near"): out[f"{tag}_ll"] = g[f"{tag}_llw"].sum() / g[f"{tag}_w"].sum().replace(0, np.nan); out[f"{tag}_acc"] = g[f"{tag}_accw"].sum() / g[f"{tag}_w"].sum().replace(0, np.nan)
    out["ece"] = [C.ece_from_bins(*bins[k]) for k in out.index]; return out.reset_index()


METRICS = ("ll", "ece", "acc", "auc", "far_auc", "far_acc", "far_ll", "near_auc", "near_ll", "htr_ll", "t13_ll", "t1x1_ll", "tc6x2_ll")


def paired(ps, cand, ref, metrics=METRICS):
    rows = []
    for split, g in ps.groupby("split"):
        a = g[g.variant == cand].set_index("seed"); b = g[g.variant == ref].set_index("seed"); common = a.index.intersection(b.index); row = {"split": split, "variant": cand, "n_seeds": len(common)}
        for m in metrics:
            if m not in a or m not in b: continue
            sgn = -1 if (m in ("ll", "ece") or m.endswith("_ll")) else 1; diff = (sgn * (a.loc[common, m] - b.loc[common, m])).dropna()
            if len(diff) < 3: continue
            try: p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
            except ValueError: p = 1.0
            bs = np.random.default_rng(0).choice(diff.to_numpy(), (2000, len(diff))).mean(1); row[m] = diff.mean(); row[m + "_lo"], row[m + "_hi"] = np.percentile(bs, [2.5, 97.5]); row[m + "_share+"] = float((diff > 0).mean()); row[m + "_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def level_table(ps, variants, cols=("auc", "acc", "ll", "ece", "far_auc", "far_ll", "near_auc", "htr_ll", "t13_ll", "t1x1_ll", "tc6x2_ll")):
    return ps[ps.variant.isin(variants)].groupby(["split", "variant"])[list(cols)].mean().reset_index()


def fit_gp_constants(cells, names):
    """Per-type temperatures of the GP's probit-corrected logits (test-oracle on the given DEVELOPMENT cells; descriptive calibrated-GP rows only)."""
    out = {}
    for nm in names:
        D = {k: [] for k in C.TYPE_NAME}; S = {k: [] for k in C.TYPE_NAME}; W = {k: [] for k in C.TYPE_NAME}
        for cell in cells:
            for rec in cell["recs"].values():
                r = rec["learners"].get(nm)
                if r is None or r["var"] is None: continue
                d = np.asarray(r["mu"], float) / np.sqrt(1 + np.pi * np.asarray(r["var"], float) / 8); t = rec["test"]
                for kk in C.TYPE_NAME: m = t["k"] == kk; D[kk].append(d[m]); S[kk].append(t["sign"][m]); W[kk].append(t["weight"][m])
        out[nm] = {}
        for kk in C.TYPE_NAME:
            d, s, w = (np.concatenate(x[kk]) for x in (D, S, W)); v = [C._wll(d, s, w, np.exp(lt)) for lt in C.LOG_T_GRID]; out[nm][kk] = float(np.exp(C.LOG_T_GRID[int(np.argmin(v))]))
    return out


def fit_gp_oof_constants(cells, names):
    """Per-type temperatures of the GP mean logits from the OUT-OF-FOLD logits of the labelled judgments of the given (development) cells; labelled judgments only, no test label."""
    out = {}
    for nm in names:
        recs = [{"oof": r["learners"][nm]["oof"]} for c in cells for r in c["recs"].values() if nm in r["learners"] and r["learners"][nm].get("oof") is not None]
        if recs: out[nm] = C.fit_prior(recs)
    return out


def md(df, fmt="{:+.4f}"): return T.to_md(df, fmt)


if __name__ == "__main__":
    OPTS = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--") and "=" in a); sys.argv = [a for a in sys.argv if not (a.startswith("--") and "=" in a)]; mode = sys.argv[1]
    if mode == "dev":
        cells = load_cells(sys.argv[2:]); gp_names = [n for n in cells[0]["learners"] if n.startswith("gp_")]; gc = fit_gp_constants(cells, gp_names); df, bins = frame(cells, gc); ps = per_seed(df, bins)
        ref = OPTS.get("ref", "lap3_k10_st4"); names = [v for v in ps.variant.unique() if v != ref]
        if OPTS.get("only"): names = [v for v in names if any(v.startswith(o) or v == o for o in OPTS["only"].split(","))]
        print(f"cells {len(cells)}, seeds {ps.seed.nunique()}; reference = {ref}\n"); print("#### levels (mean over seeds)\n"); print(md(level_table(ps, [ref, ref + "+Tcal"] + names), "{:.4f}"))
        for r in (ref, ref + "+Tcal"):
            t = pd.concat([paired(ps, v, r) for v in names if v != r]); keep = [c for c in t.columns if c in ("split", "variant", "n_seeds") or (c in ("auc", "acc", "ll", "ece", "far_auc", "far_ll", "near_auc") or c in ("auc_p", "acc_p", "ll_p"))]
            print(f"\n#### paired gain over `{r}` (positive = better; log-loss/ECE: ref - variant)\n"); print(md(t[keep]))
    if mode == "timing":
        cells = load_cells(sys.argv[2:]); rows = [{"learner": n, "budget": b, "seconds": r["seconds"], "fit_seconds": r["info"].get("fit_seconds", r["seconds"])} for c in cells for (b, d), rec in c["recs"].items() for n, r in rec["learners"].items()]
        t = pd.DataFrame(rows).groupby(["learner", "budget"])[["seconds", "fit_seconds"]].mean().reset_index(); t["all budgets"] = t.groupby("learner").seconds.transform("mean"); print("#### seconds per fit (one CPU thread; GP: fit incl. hyperparameter search, + prediction)\n"); print(md(t, "{:.2f}"))
    if mode == "variance":
        cells = load_cells(sys.argv[2:]); nm = OPTS.get("learner", "gp_cv"); rows = []
        for c in cells:
            for rec in c["recs"].values():
                r = rec["learners"].get(nm)
                if r is None: continue
                t = rec["test"]; rows.append(pd.DataFrame({"sd": np.sqrt(np.asarray(r["var"], float)), "anc": t["g_anc_min"], "lab": t["g_lab_min"], "dens": t["g_dens_max"], "k": t["k"], "z": t["sign"] * np.asarray(r["mu"], float), "zp": t["sign"] * np.asarray(r["mu"], float) / np.sqrt(1 + np.pi * np.asarray(r["var"], float) / 8), "w": t["weight"]}))
        d = pd.concat(rows); d["bin"] = np.digitize(d.anc, ANCHOR_BINS); lab = ["<=0.001", "0.001-0.01", "0.01-0.03", ">0.03"]; out = []
        for b, g in d.groupby("bin"):
            ll = np.logaddexp(0, -g.z); llp = np.logaddexp(0, -g.zp); out.append({"anchor distance": lab[b], "n judgments": len(g), "mean predictive sd of f(a)-f(b)": g.sd.mean(), "median sd": g.sd.median(), "log-loss mean score": float((g.w * ll).sum() / g.w.sum()), "log-loss probit score": float((g.w * llp).sum() / g.w.sum()), "accuracy": float((g.w * (g.z > 0)).sum() / g.w.sum())})
        print(f"#### predictive sd by distance to the nearest typed anchor ({nm}; pooled judgments of all label sets)\n"); print(md(pd.DataFrame(out), "{:.4f}"))
        print("\nSpearman correlation of the predictive sd with the distance: anchor %.3f, nearest labelled image %.3f, graph density %.3f" % tuple(float(spearmanr(d.sd, d[c]).correlation) for c in ("anc", "lab", "dens")))
        print("Spearman correlation of the sd with the absolute error-free margin |z|: %.3f; with the log-loss of the mean score: %.3f" % (float(spearmanr(d.sd, np.abs(d.z)).correlation), float(spearmanr(d.sd, np.logaddexp(0, -d.z)).correlation)))
    if mode == "grid":
        # fixed-hyperparameter grids on the development seeds: raw metrics and metrics after per-type temperatures fitted (test-oracle, diagnostic) on the same cells
        dirs = sys.argv[2:]; rows = []
        for d in dirs:
            cells = load_cells([d]); tag = Path(d).name; keys = list(cells[0]["recs"][next(iter(cells[0]["recs"]))]["grid"].keys())
            for key in keys:
                Dd = {k: [] for k in C.TYPE_NAME}; Ss = {k: [] for k in C.TYPE_NAME}; Ww = {k: [] for k in C.TYPE_NAME}
                for c in cells:
                    for rec in c["recs"].values():
                        t = rec["test"]; mu = np.asarray(rec["grid"][key][0], float)
                        for kk in C.TYPE_NAME: m = t["k"] == kk; Dd[kk].append(mu[m]); Ss[kk].append(t["sign"][m]); Ww[kk].append(t["weight"][m])
                consts = {}
                for kk in C.TYPE_NAME:
                    dd, ss, ww = (np.concatenate(x[kk]) for x in (Dd, Ss, Ww)); v = [C._wll(dd, ss, ww, np.exp(lt)) for lt in C.LOG_T_GRID]; consts[kk] = float(np.exp(C.LOG_T_GRID[int(np.argmin(v))]))
                raw, cal = [], []
                for c in cells:
                    for rec in c["recs"].values():
                        t = rec["test"]; mu = np.asarray(rec["grid"][key][0], float); s = scores(mu, t); s2 = scores(apply_t(mu, t, consts), t)
                        raw.append((s["auc"], s["acc"], s["ll"], s["far_auc"], s["ece"] if "ece" in s else np.nan)); cal.append((s2["auc"], s2["ll"], s2["far_auc"]))
                r, q = np.nanmean(np.array(raw), 0), np.nanmean(np.array(cal), 0)
                rows.append({"grid": tag, "setting (kind, c, beta, amp)": str(key), "auc": r[0], "acc": r[1], "ll": r[2], "far_auc": r[3], "auc_Tcal": q[0], "ll_Tcal": q[1], "far_auc_Tcal": q[2], **{f"T{k}": consts[k] for k in consts}})
        df = pd.DataFrame(rows).sort_values(OPTS.get("sort", "auc"), ascending=False); print(md(df.head(int(OPTS.get("top", 15))), "{:.4f}"))

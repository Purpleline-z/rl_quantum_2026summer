"""Tables of the calibration study, generated from the stage-1 cells (nothing typed by hand).

  python3 cal_tables.py dev   DIR [DIR...]                      per-variant means and paired gains over the uncalibrated candidate
  python3 cal_tables.py oracle DIR [DIR...]                     diagnostics: temperatures fitted on the TEST judgments (not a method), OOF vs test temperature
  python3 cal_tables.py confirm SPEC.json DIR [DIR...]          pre-registered confirmation (Holm, success rule)
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import nd_calibration as C

SIGN = {"ll": -1, "ece": -1, "acc": 1, "auc": 1}


def load_cells(dirs):
    return [C.load_cell(Path(f)) for d in dirs for f in sorted(glob.glob(str(Path(d) / "seed*.pkl.gz")))]


def frame(cells, variant_fn=C.variants, priors="loo", draws=None) -> tuple[pd.DataFrame, dict]:
    """One row per (seed, split, budget, draw, variant) with the test metrics, plus the per-(seed, split, variant) reliability-bin sums.  ``priors`` = 'loo': the development prior of a cell is fitted on the
    OOF logits of all OTHER seeds of the loaded cells (leave-one-seed-out); a dict is used as is; None = none."""
    rows, bins = [], {}
    for cell in cells:
        pri = priors
        if priors == "loo":
            others = [r for c in cells if c["seed"] != cell["seed"] for r in c["recs"].values()]
            pri = {"type": C.fit_prior(others), "type_test": C.fit_prior_test(others), "type_test_cf": C.fit_prior_test(others, "cf")} if len({c["seed"] for c in cells}) > 1 else None
        for (budget, draw), rec in cell["recs"].items():
            if draws is not None and draw not in draws: continue
            rec = dict(rec); rec["budget_key"] = budget
            for name, d in variant_fn(rec, pri).items():
                sc = C.scores(d, rec["test"]); b = {k: sc.pop(k) for k in ("bin_w", "bin_conf", "bin_acc")}
                rows.append({"seed": cell["seed"], "split": cell["split"], "budget": budget, "draw": draw, "variant": name, **sc})
                acc = bins.setdefault((cell["seed"], cell["split"], name), [np.zeros(C.N_BINS)] * 3); bins[(cell["seed"], cell["split"], name)] = [a + b[k] for a, k in zip(acc, ("bin_w", "bin_conf", "bin_acc"))]
    return pd.DataFrame(rows), bins


def per_seed(df, bins) -> pd.DataFrame:
    """Per (seed, split, variant): mean over budgets / draws of ll, acc, auc, per-type ll; ECE pooled over the seed's label sets; far / near log-loss pooled by weight."""
    g = df.groupby(["seed", "split", "variant"]); out = g[[c for c in df.columns if c.endswith(("_ll", "_acc")) or c in ("ll", "acc", "auc")]].mean()
    for tag in ("far", "near"): out[f"{tag}_ll"] = g[f"{tag}_llw"].sum() / g[f"{tag}_w"].sum().replace(0, np.nan); out[f"{tag}_acc"] = g[f"{tag}_accw"].sum() / g[f"{tag}_w"].sum().replace(0, np.nan); out[f"{tag}_n"] = g[f"{tag}_n"].sum()
    out["ece"] = [C.ece_from_bins(*bins[k]) for k in out.index]; return out.reset_index()


def paired(ps, cand, ref, metrics=("ll", "ece", "acc", "auc", "far_ll", "near_ll", "htr_ll", "t13_ll", "t1x1_ll", "tc6x2_ll")) -> pd.DataFrame:
    rows = []
    for split, g in ps.groupby("split"):
        a = g[g.variant == cand].set_index("seed"); b = g[g.variant == ref].set_index("seed"); common = a.index.intersection(b.index); row = {"split": split, "variant": cand, "n_seeds": len(common)}
        for m in metrics:
            if m not in a: continue
            sgn = SIGN.get(m.split("_")[-1] if m.endswith("_ll") else m, 1); diff = (sgn * (a.loc[common, m] - b.loc[common, m])).dropna()
            if not len(diff): continue
            try: p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
            except ValueError: p = 1.0
            rng = np.random.default_rng(0); bs = rng.choice(diff.to_numpy(), (2000, len(diff))).mean(1)
            row[m] = diff.mean(); row[m + "_lo"] = np.percentile(bs, 2.5); row[m + "_hi"] = np.percentile(bs, 97.5); row[m + "_share+"] = float((diff > 0).mean()); row[m + "_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def to_md(df, fmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def by_budget(df, cands, ref="raw", metrics=("ll", "acc", "auc", "far_ll")) -> pd.DataFrame:
    """Paired gain over ``ref`` per split and budget (mean over seeds of the per-seed mean over draws; log-loss: ref - cand)."""
    rows = []; df = df.assign(far_ll=df.far_llw / df.far_w.replace(0, np.nan))
    for (split, budget), g in df.groupby(["split", "budget"]):
        base = g[g.variant == ref].groupby("seed")[list(metrics)].mean()
        for c in cands:
            cur = g[g.variant == c].groupby("seed")[list(metrics)].mean(); common = cur.index.intersection(base.index); row = {"split": split, "budget": budget, "candidate": c, "n_seeds": len(common)}
            for m in metrics: row[m] = float((SIGN.get(m.split("_")[-1] if m.endswith("_ll") else m, 1) * (cur.loc[common, m] - base.loc[common, m])).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def level_table(ps, variants) -> pd.DataFrame:
    return ps[ps.variant.isin(variants)].groupby(["split", "variant"])[["ll", "ece", "acc", "auc", "far_ll", "near_ll", "htr_ll", "t13_ll", "t1x1_ll", "tc6x2_ll"]].mean().reset_index()


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items): run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


if __name__ == "__main__":
    OPTS = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--") and "=" in a); sys.argv = [a for a in sys.argv if not (a.startswith("--") and "=" in a)]
    mode = sys.argv[1]
    if mode == "dev":
        cells = load_cells(sys.argv[2:]); df, bins = frame(cells); ps = per_seed(df, bins); ref = "raw"; names = [v for v in df.variant.unique() if v != ref]
        only = OPTS.get("only", "all"); keep = [v for v in OPTS.get("variants", "").split(",") if v]; names = [v for v in names if not keep or v in keep]
        if only in ("all", "levels"): print(f"cells: {len(cells)}  seeds: {ps.seed.nunique()}\n\n#### levels (mean over seeds; log-loss / ECE lower is better)\n"); print(to_md(level_table(ps, [ref] + names), "{:.4f}"))
        t = pd.concat([paired(ps, v, ref) for v in names])
        if only in ("all", "gain"): print("\n#### gain over the uncalibrated candidate (positive = better; log-loss / ECE: raw - variant)\n")
        if only in ("all", "gain"): print(to_md(t[[c for c in t.columns if not c.endswith(("_lo", "_hi", "_share+")) and c in ("split", "variant", "n_seeds", "ll", "ll_p", "ece", "ece_p", "acc", "auc", "far_ll", "near_ll", "htr_ll", "t13_ll")]], "{:+.4f}"))

    if mode == "budget":
        cells = load_cells(sys.argv[2:]); df, bins = frame(cells); keep = [v for v in OPTS.get("variants", "T_type0_tau2,T_typeprior_tau8,CF,CF_typeprior_tau8").split(",") if v]
        print("#### paired gain over the uncalibrated candidate by budget (positive = better; log-loss: raw - variant)\n"); print(to_md(by_budget(df, keep), "{:+.4f}"))
    if mode == "oracle":
        # diagnostics only: temperature that minimises the TEST log-loss (global and per type) versus the OOF-fitted one, per budget
        cells = load_cells(sys.argv[2:]); rows = []
        for cell in cells:
            for (budget, draw), rec in cell["recs"].items():
                t, d = rec["test"], rec["d"]; oof = rec["oof"]; row = {"seed": cell["seed"], "split": cell["split"], "budget": budget, "n_dec_test": len(d), "n_dec_oof": int(np.isin(oof["outcome"], (1, -1)).sum())}
                grid = np.exp(C.LOG_T_GRID); lls = lambda msk: [float((t["weight"][msk] * np.logaddexp(0, -t["sign"][msk] * d[msk] / g)).sum() / t["weight"][msk].sum()) for g in grid]
                allm = np.ones(len(d), bool); v = lls(allm); row["logT_test_glob"] = float(np.log(grid[int(np.argmin(v))])); row["ll_gain_oracle_glob"] = v[int(np.argmin(np.abs(np.log(grid))))] - min(v); row["logT_oof_glob"] = float(np.log(C.fit_global_t(oof)))
                for kk, name in C.TYPE_NAME.items():
                    m = t["k"] == kk
                    if m.sum() >= 3: row[f"logT_test_{name}"] = float(np.log(grid[int(np.argmin(lls(m)))]))
                rows.append(row)
        df = pd.DataFrame(rows); print("#### log temperature that is optimal on the test judgments (0 = raw is already calibrated; >0 = raw is over-confident), mean over label sets\n")
        print(to_md(df.groupby(["split", "budget"])[[c for c in df.columns if c.startswith("logT") or c.startswith("n_dec") or c == "ll_gain_oracle_glob"]].mean().reset_index(), "{:+.3f}"))
        print("\ncorrelation between OOF-optimal and test-optimal global log T across label sets:", round(float(df[["logT_oof_glob", "logT_test_glob"]].corr().iloc[0, 1]), 3))

    if mode == "confirm":
        spec = json.loads(Path(sys.argv[2]).read_text()); cells = load_cells(sys.argv[3:]); priors = spec.get("priors")
        priors = {"type": {int(k): v for k, v in priors["type"].items()}} if priors else None
        df, bins = frame(cells, priors=priors if priors else ("loo" if spec.get("loo_priors") else None)); ps = per_seed(df, bins); ref = spec["reference"]; cands = spec["candidates"]
        print(f"## {spec['name']}: {ps.seed.nunique()} seeds, splits {sorted(ps.split.unique())}, reference = {ref}\n"); print("### levels (mean over seeds)\n"); print(to_md(level_table(ps, [ref] + cands), "{:.4f}"))
        tests = {}; table = {}
        for c in cands:
            t = paired(ps, c, ref); table[c] = t
            for _, r in t.iterrows():
                for m in spec["metrics"]: tests[(c, r.split, m)] = r[m + "_p"]
        adj = holm(tests); print(f"\n### tests: paired gain over `{ref}` (positive = better; Wilcoxon over seeds; Holm over the family of {len(tests)})\n"); rows = []
        for c in cands:
            for _, r in table[c].iterrows():
                for m in spec["metrics"]: rows.append({"candidate": c, "split": r.split, "metric": m, "gain": r[m], "ci_lo": r[m + "_lo"], "ci_hi": r[m + "_hi"], "share_seeds_better": r[m + "_share+"], "p": r[m + "_p"], "holm_p": adj[(c, r.split, m)]})
        print(to_md(pd.DataFrame(rows), "{:+.4f}")); print("\n### success rule\n"); verdict = []
        for c in cands:
            t = table[c].set_index("split"); sig = sum(adj[(c, sp, m)] < spec["alpha"] and t.loc[sp, m] > 0 for sp in t.index for m in spec["metrics"]); llpos = all(t.loc[sp, "ll"] > 0 for sp in t.index)
            side = all(t.loc[sp, "auc"] >= -spec["auc_margin"] and t.loc[sp, "acc"] >= -spec["acc_margin"] for sp in t.index); ok = bool(llpos and sig >= spec["need_sig"] and side)
            verdict.append({"candidate": c, "log-loss gain > 0 in both splits": llpos, f"Holm-significant positive tests (of {2 * len(spec['metrics'])})": sig, "AUC >= -%g and acc >= -%g in both splits" % (spec["auc_margin"], spec["acc_margin"]): side, "SUCCESS": ok})
        print(to_md(pd.DataFrame(verdict)))
        print("\n### side metrics (descriptive; same paired gains, positive = better; NOT in the success rule)\n")
        side_rows = []
        for c in cands:
            for _, r in table[c].iterrows(): side_rows.append({"candidate": c, "split": r.split, **{m: r.get(m, np.nan) for m in ("acc", "auc", "far_ll", "near_ll", "t1x1_ll", "tc6x2_ll", "t13_ll", "htr_ll")}, "far_ll_p(uncorrected)": r.get("far_ll_p", np.nan)})
        print(to_md(pd.DataFrame(side_rows), "{:+.4f}")); print("\n### paired gain by budget (descriptive; positive = better)\n"); print(to_md(by_budget(df, cands, ref), "{:+.4f}"))
    if mode == "learners":
        # calibrated comparison of learners: each learner's logits get the SAME calibrator (fitted on that learner's own OOF logits / development prior), then paired differences cand - ref
        spec = json.loads(Path(sys.argv[2]).read_text()); frames = []
        for lname, dirs in spec["learners"].items():
            pri = spec.get("priors", {}).get(lname, "loo"); pri = {"type": {int(k): v for k, v in pri["type"].items()}} if isinstance(pri, dict) else pri; cells = load_cells(dirs); df, bins = frame(cells, priors=pri, draws=spec.get("draws")); ps = per_seed(df, bins); ps = ps[ps.variant.isin(spec["variants"])].copy(); ps["variant"] = lname + "|" + ps.variant; frames.append(ps)
        ps = pd.concat(frames, ignore_index=True); cand, ref = spec["candidate"], spec["reference"]
        print("#### levels (mean over seeds)\n"); print(to_md(level_table(ps, sorted(ps.variant.unique())), "{:.4f}"))
        print(f"\n#### paired difference `{cand}` minus `{ref}` under the same calibrator (positive = candidate better; log-loss / ECE: ref - cand)\n")
        t = pd.concat([paired(ps, f"{cand}|{v}", f"{ref}|{v}") for v in spec["variants"]]); t["variant"] = [x.split("|")[1] for x in t.variant]
        print(to_md(t[[c for c in t.columns if not c.endswith(("_lo", "_hi", "_share+")) and c in ("split", "variant", "n_seeds", "ll", "ll_p", "ece", "acc", "auc", "auc_p", "far_ll", "near_ll", "htr_ll", "t13_ll", "t1x1_ll", "tc6x2_ll")]], "{:+.4f}"))

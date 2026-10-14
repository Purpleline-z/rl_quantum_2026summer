"""Descriptive analyses of the honest-inference study (PREREGISTRATION_HI1, 'Descriptive analyses'): coverage of the groups by the test sets, per-group gains (leave-group-out view), variance
decomposition of the per-judgment gain (crossed random effects), bootstrap standard errors under different resampling schemes, effective number of independent test units, power analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import hi_infer as H
import hi_analyze as A


def coverage(blk: H.Block) -> pd.DataFrame:
    """How often every group is in a held-out set: number of (seed, split) test sets (and of cells) containing the group."""
    ss = pd.DataFrame({"seed": blk.cells.seed.to_numpy()[blk.cell_of], "split": blk.cells.split.to_numpy()[blk.cell_of], "g": blk.grp}).drop_duplicates()
    n_sets = ss.groupby("g").size().reindex(range(blk.n_groups), fill_value=0); n_cells = pd.Series(np.bincount(blk.grp, minlength=blk.n_groups), index=range(blk.n_groups))   # judgments, not cells: see per_cell below
    cl = blk.clusters.copy(); cl["n_testsets"] = n_sets.to_numpy(); cl["n_test_judgments"] = n_cells.to_numpy(); return cl


def group_gains(blk: H.Block, cand: str, ref: str, split: str) -> pd.DataFrame:
    """Per group (leave-group-out view): mean accuracy / log-loss gain over all its held-out judgments (all cells of the split), and the AUC contribution per appearance (mean over groups, weighted by appearances, = the AUC gain)."""
    cm = (blk.cells.split == split).to_numpy(); mask_j = cm[blk.cell_of]; rows = pd.DataFrame({"g": np.arange(blk.n_groups)}).set_index("g")
    ma, mr = blk.metric_arrays(cand), blk.metric_arrays(ref); d_acc = ma["acc"] - mr["acc"]; d_ll = mr["ll"] - ma["ll"]
    df = pd.DataFrame({"g": blk.grp[mask_j], "acc": d_acc[mask_j], "ll": d_ll[mask_j], "w": blk.wt[mask_j]}); agg = df.groupby("g").apply(lambda x: pd.Series({"acc_gain": np.average(x.acc, weights=x.w), "ll_gain": np.average(x.ll, weights=x.w), "n_judgments": len(x)}), include_groups=False)
    des = H.Design(blk, cand, ref, "auc", cell_mask=cm); contrib = np.zeros(blk.n_groups); app = np.zeros(blk.n_groups)
    for r, ids, a in des.contributions_by_cell(): np.add.at(contrib, ids, a * len(ids)); np.add.at(app, ids, 1)
    rows = rows.join(agg); rows["auc_gain_per_appearance"] = np.where(app > 0, contrib / np.maximum(app, 1), np.nan); rows["n_appearances"] = app
    return rows.dropna(subset=["n_judgments"])


def group_sign_summary(gg: pd.DataFrame) -> dict:
    out = {"n_groups": len(gg)}
    for col in ("acc_gain", "ll_gain", "auc_gain_per_appearance"):
        x = gg[col].dropna(); pos, neg = int((x > 1e-12).sum()), int((x < -1e-12).sum()); out.update({f"{col}_pos": pos, f"{col}_neg": neg, f"{col}_zero": len(x) - pos - neg, f"{col}_sign_p": float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0, f"{col}_mean": float(x.mean()), f"{col}_sd": float(x.std(ddof=1))})
    return out


def se_schemes(blk: H.Block, cand: str, ref: str, split: str, metric: str, B: int = 2000, seed: int = 0) -> dict:
    """Bootstrap standard error of the estimate under different resampling schemes (draws only; seeds only; pair groups; image-cluster units; coarse blocks)."""
    cm = (blk.cells.split == split).to_numpy(); des = H.Design(blk, cand, ref, metric, cell_mask=cm); out = {"estimate": des.estimate()}
    if metric != "auc":      # per-judgment sd of the gain (confidence-weighted) -> effective number of independent judgments = sd^2 / se_group^2
        mj = cm[blk.cell_of]; ma, mr = blk.metric_arrays(cand), blk.metric_arrays(ref); y = (ma["acc"] - mr["acc"]) if metric == "acc" else (mr["ll"] - ma["ll"]); w = blk.wt[mj]; yy = y[mj]; mu = np.average(yy, weights=w)
        out["sd_judgment"] = float(np.sqrt(np.average((yy - mu) ** 2, weights=w))); out["n_judgment_appearances"] = int(mj.sum()); out["n_unique_judgments"] = int(len(np.unique(np.stack([blk.grp[mj], blk.k[mj], blk.sign[mj]], 1), axis=0)))
    out["se_draws"] = float(np.std(H.draw_bootstrap(des, B, seed))); out["se_seeds"] = float(np.std(H.seed_resample(des, B, seed)))
    for col in ("unit_g", "unit_c0.03", "unit_b84", "unit_b42", "unit_b21"): out["se_" + col.replace("unit_", "")] = float(np.nanstd(H.group_bootstrap(des, A.unit_of(blk, col), B=B, seed=seed)))
    return out


def variance_components(blk: H.Block, cand: str, ref: str, split: str, metric: str, max_rows: int = 60000, seed: int = 0) -> dict:
    """Crossed random effects of the per-judgment gain (accuracy: dcorrect, log-loss: ll_ref - ll_cand) aggregated to (cell, group) means: unit (cluster-pair unit c0.03), group within unit, seed, cell (label draw / budget
    / split specific), residual (judgment level).  statsmodels MixedLM with variance components; returns variances and the share of each."""
    import statsmodels.formula.api as smf
    cm = (blk.cells.split == split).to_numpy(); mj = cm[blk.cell_of]; ma, mr = blk.metric_arrays(cand), blk.metric_arrays(ref); y = (ma["acc"] - mr["acc"]) if metric == "acc" else (mr["ll"] - ma["ll"])
    df = pd.DataFrame({"cell": blk.cell_of[mj], "g": blk.grp[mj], "y": y[mj]}); df = df.groupby(["cell", "g"], as_index=False).agg(y=("y", "mean"), n=("y", "size"))
    cells = blk.cells.reset_index(); df["seed"] = cells.seed.to_numpy()[df.cell]; unit = A.unit_of(blk, "unit_c0.03"); df["unit"] = unit[df.g]
    if len(df) > max_rows: df = df.sample(max_rows, random_state=seed)
    df["one"] = 1; vc = {"unit": "0 + C(unit)", "group": "0 + C(g)", "seed": "0 + C(seed)", "cell": "0 + C(cell)"}
    fit = smf.mixedlm("y ~ 1", df, groups="one", vc_formula=vc, re_formula="0").fit(reml=True, method="lbfgs"); names = list(fit.model.exog_vc.names); v = dict(zip(names, np.asarray(fit.vcomp))); v["residual"] = float(fit.scale); tot = sum(v.values())
    return {"intercept": float(fit.fe_params["Intercept"]), "n_rows": len(df), "converged": bool(fit.converged), **{f"var_{k}": float(x) for k, x in v.items()}, **{f"share_{k}": float(x / tot) for k, x in v.items()}}


def power_table(res: pd.DataFrame, n_groups_total: int = 168) -> pd.DataFrame:
    """Power analysis from the group-bootstrap standard errors (pooled rows): minimal detectable true gain at 80 % power, alpha = 0.05 two-sided (normal approximation: (1.96 + 0.8416) * SE) and the total
    number of independent units needed for 80 % power at the observed gain and at half of it (units in design * (2.8016 SE / gain)^2) and the implied number of new groups beyond the groups of the design."""
    z = 1.959964 + 0.841621; rows = []
    for r in res.itertuples():
        se = r.se; units = r.n_units; mde = z * se; row = dict(comp=r.comp, split=r.split, metric=r.metric, stratum=r.stratum, est=r.est, se=se, mde80=mde, units=units)
        for tag, delta in (("obs", abs(r.est)), ("half", abs(r.est) / 2)):
            n_tot = units * (z * se / delta) ** 2 if delta > 1e-12 else np.inf; row[f"units_needed_{tag}"] = n_tot; row[f"new_units_{tag}"] = max(0.0, n_tot - units) if np.isfinite(n_tot) else np.inf
        rows.append(row)
    return pd.DataFrame(rows)

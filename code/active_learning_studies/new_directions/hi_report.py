"""Script-generated tables and figures of the honest-inference study (all numbers of RESULTS_honest_inference.md come from here).
  python3 hi_report.py [--analysis results/hi/analysis] [--out ../pair_disjoint_not_image_disjoint/graph_exploration/HI_PACKAGE]
Reads the CSVs written by hi_pipeline.py; applies the re-assessment rule frozen in PREREGISTRATION_HI1.md (hi_analyze.assess)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hi_analyze as A

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE.parent / "pair_disjoint_not_image_disjoint" / "graph_exploration" / "HI_PACKAGE"
LABEL = {c[0]: c[3] for c in A.COMPS}
# (claim id, claim text, original status of the paper package, comparison, kind): kind = pooled (re-assessed with the frozen rule) | htr (exploratory HTR reading) | strata | blocked | control
CLAIMS = [("L1", "aw8 (anchor weight 0.25 -> 8) beats the default head", "CONFIRMED + replicated", "C1", "pooled"),
          ("L2", "aw8_all (all ideal images as anchors) beats the default head", "CONFIRMED", "C2", "pooled"),
          ("L2b", "aw8_all vs aw8", "DESCRIPTIVE", "C2b", "pooled"),
          ("L3", "Laplacian kNN-graph regulariser (lap3_k10) beats aw8_all", "CONFIRMED + replicated", "C3", "pooled"),
          ("L4", "4x training steps (lap3_k10_st4) beats lap3_k10", "CONFIRMED + replicated", "C4", "pooled"),
          ("L5", "BEST: lap3_k10_st4 beats the tuned graph-free control", "CONFIRMED + replicated", "C5", "pooled"),
          ("L6", "whole recipe (lap3_k10_st4) vs aw8_all", "DESCRIPTIVE", "C6", "pooled"),
          ("L7", "graph-free schedule tuning alone vs aw8_all", "DESCRIPTIVE", "C7", "pooled"),
          ("L8", "real graph beats shuffled-graph control (a low bar)", "PRE-REG. control", "C8", "control"),
          ("L9", "RT13-off variant beats lap3_k10 (and, new: vs lap3_k10_st4)", "CONFIRMED + replicated", "C9", "pooled"),
          ("C10", "per-type temperature calibration improves log-loss of lap3_k10_st4 (new)", "(calibration branch)", "C10", "calib"),
          ("C11", "RT13-off vs lap3_k10_st4 (new)", "(not tested before)", "C11", "pooled"),
          ("L13", "aw8_all improves HTR vs default head", "NOT CONFIRMED", "C2", "htr"),
          ("S5", "HTR: st4 vs tuned graph-free control", "mixed", "C5", "htr"),
          ("S6", "HTR: st4 vs lap3_k10", "mixed", "C4", "htr"),
          ("R1", "lap3_k10_st4 vs tuned control: gain concentrated in anchor-near judgments", "AUDIT (red-team dev)", "C5", "strata"),
          ("R2", "aw8 / aw8_all gain is not an anchor-copy artefact", "AUDIT (red-team dev)", "C2", "strata"),
          ("R3", "time-blocked split keeps the gain (lap3_k10_st4 vs tuned control)", "AUDIT (red-team dev)", "C5", "blocked")]
NOT_REASSESSED = [("L10", "uniform lambda 10 beats lap3_k10", "not a stored learner in these runs"), ("L11", "Laplacian + type-posterior term beats aw8_all", "learner not re-run"), ("L12", "GP learner beats the anchor-weighted MLP head", "learner not re-run"),
                  ("D1", "long schedule does not help graph-free aw8_all", "development observation; learner not re-run"), ("D2", "graph-free consistency regulariser comes close", "development observation; learner not re-run"),
                  ("G1", "gain tracks graph quality (noise dose-response)", "graph-noise learners not re-run"), ("G2", "10% random rewiring destroys the gain", "development observation; not re-run"), ("G3", "shuffled graph far worse than no graph (lambda 10)", "lambda-10 shuffled control not re-run; C8 uses the st4 shuffled control"),
                  ("G4", "more unlabelled frames: gain saturates", "pool-size runs not re-run"), ("G5", "harsh splits (held-out run / temperature)", "fixed-test-set runs not re-run; see time-blocked contexts (R3) for the nearest check"), ("G6", "paper's type-accuracy endpoint (28 ideal images)", "different endpoint, no per-judgment predictions"),
                  ("S1", "vopt_u beats random (default head)", "selection runs not re-run"), ("S2", "vopt_u beats random (aw8 head)", "selection runs not re-run"), ("S3", "4 selection rules vs random on lap3_k10", "selection runs not re-run"), ("S4", "HTR-directed selection", "selection runs not re-run"),
                  ("D3-D?", "mechanism / ablation observations of the graph-mechanism branch", "not re-run")]


def f3(x): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"
def ci(est, lo, hi): return f"{f3(est)} [{f3(lo)}, {f3(hi)}]"
def ratio(a, b): return a / b if b else float("nan")
def pf(p): return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else (f"{p:.4f}" if p >= 1e-4 else "<1e-4")


def rd(analysis: Path, name: str):
    p = analysis / name; return pd.read_csv(p) if p.exists() else None


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False): lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def row(df, comp, split, metric, stratum="all"):
    s = df[(df.comp == comp) & (df.split == split) & (df.metric == metric) & (df.stratum == stratum)]
    return s.iloc[0] if len(s) else None


def stratum_cell(df, comp, stratum, metric="auc"):
    out, ok = [], []
    for sp in ("A", "B"):
        r = row(df, comp, sp, metric, stratum) if df is not None else None
        if r is None: return "n/a", None
        out.append(ci(r.est, r.lo, r.hi)); ok.append(r.lo > 0)
    return " / ".join(out), all(ok)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", type=Path, default=HERE / "results" / "hi" / "analysis"); ap.add_argument("--out", type=Path, default=OUT_DEFAULT); a = ap.parse_args()
    tab = a.out / "tables"; fig = a.out / "figures"; tab.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)
    main_conf = rd(a.analysis, "main_conf.csv"); strata_conf = rd(a.analysis, "strata_conf.csv"); blocked = rd(a.analysis, "blocked_conf.csv"); numbers: dict = {}
    if main_conf is None: raise SystemExit("main_conf.csv missing: run hi_pipeline.py first")
    # ---------------------------------------------------------------- assessment (frozen rule) pooled + replicates
    ass = A.assess(main_conf).set_index("comp"); rep = {}
    for nm in ("conf1", "conf2", "dev"):
        d = rd(a.analysis, f"main_{nm}.csv"); rep[nm] = A.assess(d).set_index("comp")["label"] if d is not None else None
    # ---------------------------------------------------------------- main table (every comparison x split x metric)
    cols = []
    for r in main_conf.itertuples():
        cols.append(dict(comp=r.comp, comparison=LABEL[r.comp], split=r.split, metric=r.metric, estimate=r.est, group_lo=r.lo, group_hi=r.hi, group_se=r.se, seed_lo=r.seed_lo, seed_hi=r.seed_hi, seed_wilcoxon_p=r.seed_wilcoxon_p, group_perm_p=r.perm_p,
                         cluster_lo=getattr(r, "lo_c0.03", np.nan) if hasattr(r, "lo_c0.03") else np.nan))
    mt = main_conf.copy(); mt["width_ratio_group_over_seed"] = [ratio(r.hi - r.lo, r.seed_hi - r.seed_lo) for r in mt.itertuples()]; mt.to_csv(tab / "main_all.csv", index=False)
    lines = ["| comparison | split | metric | estimate | seed-level 95% CI | seed Wilcoxon p | group-level 95% CI | group randomisation p | cluster (c0.03) 95% CI | block (b42) 95% CI | group CI width / seed CI width |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in main_conf.itertuples():
        g = lambda k: getattr(r, k.replace(".", "_"), np.nan)
        lines.append(f"| {r.comp} {LABEL[r.comp]} | {r.split} | {r.metric} | {f3(r.est)} | [{f3(r.seed_lo)}, {f3(r.seed_hi)}] | {pf(r.seed_wilcoxon_p)} | [{f3(r.lo)}, {f3(r.hi)}] | {pf(r.perm_p)} | [{f3(g('lo_c0.03'))}, {f3(g('hi_c0.03'))}] | [{f3(g('lo_b42'))}, {f3(g('hi_b42'))}] | {ratio(r.hi - r.lo, r.seed_hi - r.seed_lo):.1f} |")
    (tab / "main_table.md").write_text("Pooled confirmation 1 + 2 (70 seeds), standard endpoint, all judgments. Intervals are percentile 95% bootstrap intervals; p-values: seed-level Wilcoxon vs exact group-level randomisation test (unadjusted).\n\n" + "\n".join(lines) + "\n")
    # ---------------------------------------------------------------- Holm / assessment table
    f1 = main_conf[(main_conf.comp.isin(A.FAMILY1)) & (main_conf.stratum == "all") & (main_conf.metric.isin(["auc", "acc"])) & (main_conf.split.isin(["A", "B"]))].copy(); f1["holm_p"] = A.holm(f1.perm_p.to_numpy())
    f1[["comp", "split", "metric", "est", "lo", "hi", "perm_p", "holm_p", "seed_wilcoxon_p"]].to_csv(tab / "family1_holm.csv", index=False)
    cal = main_conf[(main_conf.comp == "C10") & (main_conf.metric == "ll") & (main_conf.split.isin(["A", "B"]))].copy(); cal["holm_p_F2"] = A.holm(cal.perm_p.to_numpy()); cal.to_csv(tab / "family2_calibration.csv", index=False)
    # ---------------------------------------------------------------- corrected claims table
    rows = []
    for cid, text, orig, comp, kind in CLAIMS:
        d = dict(id=cid, claim=text, paper_status=orig, comparison=comp, kind=kind)
        if kind == "pooled":
            r = ass.loc[comp]; d.update(honest_label=r.label, auc_A=ci(r.auc_A, r.auc_A_lo, r.auc_A_hi), auc_B=ci(r.auc_B, r.auc_B_lo, r.auc_B_hi), acc_A=ci(r.acc_A, r.acc_A_lo, r.acc_A_hi), acc_B=ci(r.acc_B, r.acc_B_lo, r.acc_B_hi), holm_p_A=pf(r.auc_A_holm_p), holm_p_B=pf(r.auc_B_holm_p))
            sw = [row(main_conf, comp, s, "auc") for s in ("A", "B")]; d["seed_wilcoxon_p_auc"] = " / ".join(pf(x.seed_wilcoxon_p) for x in sw); d["group_ci_vs_seed_ci_width"] = " / ".join(f"{ratio(x.hi - x.lo, x.seed_hi - x.seed_lo):.1f}x" for x in sw)
            d["robust_cluster_c0.03"] = all(x["lo_c0.03"] > 0 for x in sw); d["robust_block_b42"] = all(x["lo_b42"] > 0 for x in sw)
            d["conf1_label"] = rep["conf1"].get(comp) if rep["conf1"] is not None else None; d["conf2_label"] = rep["conf2"].get(comp) if rep["conf2"] is not None else None; d["dev_label"] = rep["dev"].get(comp) if rep["dev"] is not None else None
            d["replicated"] = bool(d["conf1_label"] == d["conf2_label"] == r.label)
            for st in ("near", "far03", "far01"): d[f"auc_{st}"], d[f"{st}_excl0"] = stratum_cell(strata_conf, comp, st)
            d["blocked_auc"] = ci(*(lambda x: (x.est, x.lo, x.hi))(row(blocked, comp, "blocked", "auc"))) if blocked is not None and row(blocked, comp, "blocked", "auc") is not None else "n/a"
        elif kind == "control":
            sw = [row(main_conf, comp, s, "auc") for s in ("A", "B")]; ok = all(x.lo > 0 for x in sw); d.update(honest_label="CONTROL PASSED" if ok else "CONTROL NOT PASSED", auc_A=ci(sw[0].est, sw[0].lo, sw[0].hi), auc_B=ci(sw[1].est, sw[1].lo, sw[1].hi))
            for s, x in zip("AB", [row(main_conf, comp, s, "acc") for s in "AB"]): d[f"acc_{s}"] = ci(x.est, x.lo, x.hi)
        elif kind == "calib":
            sw = [row(main_conf, comp, s, "ll") for s in "AB"]; ok = all(x.lo > 0 for x in sw) and all(cal.set_index("split").loc[s, "holm_p_F2"] < 0.05 for s in "AB"); d.update(honest_label="SUPPORTED" if ok else "NOT SUPPORTED", logloss_A=ci(sw[0].est, sw[0].lo, sw[0].hi), logloss_B=ci(sw[1].est, sw[1].lo, sw[1].hi),
                                                                                           holm_p_A=pf(cal.set_index("split").loc["A", "holm_p_F2"]), holm_p_B=pf(cal.set_index("split").loc["B", "holm_p_F2"]))
            for s, x in zip("AB", [row(main_conf, comp, s, "auc") for s in "AB"]): d[f"auc_{s}"] = ci(x.est, x.lo, x.hi)
        elif kind == "htr":
            ra = [row(strata_conf, comp, s, "acc", "htr") for s in "AB"] if strata_conf is not None else [None, None]; rl = [row(strata_conf, comp, s, "ll", "htr") for s in "AB"] if strata_conf is not None else [None, None]
            if any(x is None for x in ra + rl): d["honest_label"] = "n/a"
            else:
                ok = all(x.lo > 0 for x in ra); weak = all(x.est > 0 for x in ra); d.update(honest_label=("SUPPORTED (HTR accuracy, exploratory reading)" if ok else ("WEAKER" if weak else "UNSUPPORTED")), acc_A=ci(ra[0].est, ra[0].lo, ra[0].hi), acc_B=ci(ra[1].est, ra[1].lo, ra[1].hi), logloss_A=ci(rl[0].est, rl[0].lo, rl[0].hi), logloss_B=ci(rl[1].est, rl[1].lo, rl[1].hi))
        elif kind == "strata":
            for st in ("near", "mid", "far03", "far01"): d[f"auc_{st}"], d[f"{st}_excl0"] = stratum_cell(strata_conf, comp, st)
            near, far = d.get("near_excl0"), d.get("far03_excl0")
            d["honest_label"] = ("gain present in both strata (interval excludes 0 in near and far03)" if near and far else ("gain only in anchor-near" if near and far is False else ("gain only in anchor-far" if far and near is False else ("neither stratum excludes 0" if near is False and far is False else "n/a"))))
            if comp == "C2":
                d1 = {}; d["auc_far03_C1"], _ = stratum_cell(strata_conf, "C1", "far03"); d["auc_near_C1"], _ = stratum_cell(strata_conf, "C1", "near")
        elif kind == "blocked":
            if blocked is None: d["honest_label"] = "n/a"
            else:
                r = row(blocked, comp, "blocked", "auc"); d.update(honest_label="gain kept (interval excludes 0)" if r.lo > 0 else "interval includes 0", blocked_auc=ci(r.est, r.lo, r.hi), blocked_acc=ci(*(lambda x: (x.est, x.lo, x.hi))(row(blocked, comp, "blocked", "acc"))))
        rows.append(d)
    ct = pd.DataFrame(rows); ct.to_csv(tab / "claims_corrected.csv", index=False)
    L = ["| ID | claim | paper status | honest label (frozen rule) | ΔAUC A (group 95% CI) | ΔAUC B | Δacc A / B (point) | seed Wilcoxon p (A / B) | group Holm p (A / B) | robust: cluster / block | replicated (conf1, conf2) | anchor-near ΔAUC | anchor-far (>0.03) ΔAUC | time-blocked ΔAUC |", "|" + "|".join("---" for _ in range(14)) + "|"]
    for d in rows:
        if d["kind"] == "pooled":
            L.append(f"| {d['id']} | {d['claim']} | {d['paper_status']} | **{d['honest_label']}** | {d['auc_A']} | {d['auc_B']} | {d['acc_A'].split(' ')[0]} / {d['acc_B'].split(' ')[0]} | {d['seed_wilcoxon_p_auc']} | {d['holm_p_A']} / {d['holm_p_B']} | {'yes' if d['robust_cluster_c0.03'] else 'no'} / {'yes' if d['robust_block_b42'] else 'no'} | {'yes' if d['replicated'] else 'no'} ({d['conf1_label']}, {d['conf2_label']}) | {d['auc_near']} | {d['auc_far03']} | {d['blocked_auc']} |")
        elif d["kind"] == "control":
            L.append(f"| {d['id']} | {d['claim']} | {d['paper_status']} | **{d['honest_label']}** | {d['auc_A']} | {d['auc_B']} | {d['acc_A'].split(' ')[0]} / {d['acc_B'].split(' ')[0]} | | | | | | | |")
        elif d["kind"] == "calib":
            L.append(f"| {d['id']} | {d['claim']} | {d['paper_status']} | **{d['honest_label']}** | log-loss A {d['logloss_A']} | log-loss B {d['logloss_B']} | (AUC {d['auc_A'].split(' ')[0]} / {d['auc_B'].split(' ')[0]}) | | {d['holm_p_A']} / {d['holm_p_B']} (F2, log-loss) | | | | | |")
        elif d["kind"] == "htr":
            L.append(f"| {d['id']} | {d['claim']} | {d['paper_status']} | **{d['honest_label']}** | HTR acc A {d.get('acc_A', 'n/a')} | HTR acc B {d.get('acc_B', 'n/a')} | HTR log-loss A / B: {d.get('logloss_A', 'n/a')} / {d.get('logloss_B', 'n/a')} | | | | | | | |")
        elif d["kind"] == "strata":
            L.append(f"| {d['id']} | {d['claim']} | {d['paper_status']} | **{d['honest_label']}** | | | | | | | | {d['auc_near']} | {d['auc_far03']} | |")
        else:
            L.append(f"| {d['id']} | {d['claim']} | {d['paper_status']} | **{d['honest_label']}** | | | | | | | | | | {d.get('blocked_auc', 'n/a')} |")
    for cid, text, why in NOT_REASSESSED: L.append(f"| {cid} | {text} | (paper package) | not re-assessed | | | | | | | | | | reason: {why} |")
    (tab / "claims_corrected.md").write_text("Corrected claims table (script-generated by hi_report.py from results/hi/analysis; pooled confirmation 1 + 2, 70 seeds, 168 groups). Label rule frozen in PREREGISTRATION_HI1.md: SUPPORTED = group-level AUC interval excludes 0 in splits A and B, accuracy gain positive in both, Holm-adjusted (F1, 40 tests) randomisation p < 0.05 in both; WEAKER = AUC gain positive in both splits but not SUPPORTED; UNSUPPORTED = AUC gain <= 0 in a split. ΔAUC cells: estimate [group-bootstrap 95% CI]. Near / far strata and time-blocked: exploratory, uncorrected.\n\n" + "\n".join(L) + "\n")
    # label counts
    cnt = ct[ct.kind == "pooled"].honest_label.value_counts().to_dict(); numbers["label_counts"] = cnt
    # ---------------------------------------------------------------- strata table (all comparisons)
    if strata_conf is not None:
        s = strata_conf[strata_conf.metric == "auc"].copy(); s["cell"] = [ci(r.est, r.lo, r.hi) for r in s.itertuples()]
        piv = s.pivot_table(index=["comp", "stratum"], columns="split", values="cell", aggfunc="first").reset_index(); piv.insert(1, "comparison", piv.comp.map(LABEL)); order = {st: i for i, st in enumerate(["near", "mid", "far03", "far01"])}; piv = piv[piv.stratum.isin(order)].sort_values(["comp", "stratum"], key=lambda c: c.map(order) if c.name == "stratum" else c)
        (tab / "strata_auc.md").write_text("ΔAUC by anchor stratum (pooled confirmation, group bootstrap B = 4000, 95% percentile CI; near: anc <= 0.001; mid: 0.001 < anc <= 0.03; far03: anc > 0.03; far01: anc > 0.01). Exploratory.\n\n" + md_table(piv) + "\n")
        n_by = {}; frac = {}
        s.to_csv(tab / "strata_auc.csv", index=False)
    # ---------------------------------------------------------------- blocked
    if blocked is not None:
        b = blocked[blocked.stratum == "all"].copy(); b["cell"] = [ci(r.est, r.lo, r.hi) for r in b.itertuples()]; piv = b.pivot_table(index="comp", columns="metric", values="cell", aggfunc="first").reset_index(); piv.insert(1, "comparison", piv.comp.map(LABEL)); piv["seed-level Wilcoxon p (AUC)"] = piv.comp.map(b[b.metric == "auc"].set_index("comp").seed_wilcoxon_p.map(pf))
        piv["group randomisation p (AUC)"] = piv.comp.map(b[b.metric == "auc"].set_index("comp").perm_p.map(pf)); (tab / "blocked.md").write_text("Time-blocked contexts (seeds 11330-11399, one draw per budget), group bootstrap B = 4000, 95% CI. Gain = candidate - reference. Exploratory.\n\n" + md_table(piv) + "\n")
    # ---------------------------------------------------------------- SE schemes, coverage, group signs, variance
    for nm in ("conf",):
        se = rd(a.analysis, f"se_schemes_{nm}.csv")
        if se is not None:
            se["effective_independent_judgments"] = (se.sd_judgment ** 2) / (se.se_g ** 2) if "sd_judgment" in se else np.nan; se["se_ratio_groups_over_seeds"] = se.se_g / se.se_seeds; se["se_ratio_cluster_over_group"] = se["se_c0.03"] / se.se_g; se["se_ratio_b42_over_group"] = se.se_b42 / se.se_g
            se.to_csv(tab / "se_schemes.csv", index=False); keep = ["comp", "split", "metric", "estimate", "se_draws", "se_seeds", "se_g", "se_c0.03", "se_b84", "se_b42", "se_b21", "se_ratio_groups_over_seeds", "effective_independent_judgments"]
            (tab / "se_schemes.md").write_text("Bootstrap standard error of the paired gain under different resampling units (pooled confirmation). se_draws: label draws only; se_seeds: seeds only (the unit of the paper package's Wilcoxon p-values); se_g: pair groups; se_c0.03: image-cluster-pair units; se_b84/b42/b21: coarse look-alike blocks. effective_independent_judgments = (per-judgment sd of the gain)^2 / se_g^2 (accuracy / log-loss only).\n\n" + md_table(se[[c for c in keep if c in se]], "{:.4g}") + "\n")
        cov = rd(a.analysis, f"coverage_{nm}.csv")
        if cov is not None:
            numbers.update(cov_min_testsets=int(cov.n_testsets.min()), cov_median_testsets=float(cov.n_testsets.median()), cov_groups=int((cov.n_test_judgments > 0).sum()), cov_all_groups=len(cov)); (tab / "coverage_summary.md").write_text(f"Groups with at least one held-out appearance: {numbers['cov_groups']} of {len(cov)}. Number of (seed, split) test sets containing a group: min {cov.n_testsets.min()}, median {cov.n_testsets.median():.0f}, max {cov.n_testsets.max()}.\n")
        gs = rd(a.analysis, f"group_signs_{nm}.csv")
        if gs is not None: gs.to_csv(tab / "group_signs.csv", index=False); (tab / "group_signs.md").write_text("Leave-group-out view: number of the 168 groups whose own held-out judgments gain (mean over their cells) under each comparison; two-sided sign test of the group-level sign.\n\n" + md_table(gs[["comp", "split", "n_groups", "acc_gain_pos", "acc_gain_neg", "acc_gain_zero", "acc_gain_sign_p", "ll_gain_pos", "ll_gain_neg", "ll_gain_sign_p", "auc_gain_per_appearance_pos", "auc_gain_per_appearance_neg", "auc_gain_per_appearance_sign_p"]], "{:.4g}") + "\n")
        vc = rd(a.analysis, f"variance_components_{nm}.csv")
        if vc is not None and len(vc): vc.to_csv(tab / "variance_components.csv", index=False); sh = [c for c in vc.columns if c.startswith("share_")]; (tab / "variance_components.md").write_text("Variance decomposition of the per-judgment gain aggregated to (cell, group) means; crossed random effects (REML): image-cluster unit (c0.03), group, seed, cell (label draw x budget x split), residual. Shares of the total variance.\n\n" + md_table(vc[["comp", "split", "metric", "intercept", "n_rows"] + sh], "{:.4g}") + "\n")
    # ---------------------------------------------------------------- power
    try:
        import hi_extra as X
        pw = X.power_table(main_conf[(main_conf.stratum == "all") & (main_conf.split.isin(["A", "B"]))]); pw.to_csv(tab / "power.csv", index=False)
        key = pw[pw.comp.isin(["C2", "C3", "C4", "C5", "C9"]) & pw.metric.isin(["auc", "acc"])]
        (tab / "power.md").write_text("Power at the group level (normal approximation, alpha = 0.05 two-sided, power 80%): mde80 = 2.80 x group-bootstrap SE; units_needed = (units in the design) x (2.80 SE / gain)^2 for the observed gain and for half of it; new_units = units_needed - units in the design (pair groups, assuming the same per-group variance and heterogeneity). Pooled confirmation.\n\n" + md_table(key[["comp", "split", "metric", "est", "se", "mde80", "units", "units_needed_obs", "new_units_obs", "units_needed_half", "new_units_half"]], "{:.4g}") + "\n")
    except Exception as e: print("power table skipped:", repr(e)[:120])
    # ---------------------------------------------------------------- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ids = [c for c in ["C2", "C3", "C4", "C5", "C9", "C11", "C7", "C6"] if c in set(main_conf.comp)]; fig1, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(ids) + 1.8), sharey=True)
    for ax, sp in zip(axes, "AB"):
        for i, c in enumerate(ids):
            r = row(main_conf, c, sp, "auc"); y = -i
            ax.plot([r.seed_lo, r.seed_hi], [y + 0.18, y + 0.18], color="tab:gray", lw=3, label="seed-level (paper package)" if i == 0 else None); ax.plot([r.lo, r.hi], [y, y], color="tab:blue", lw=3, label="pair groups" if i == 0 else None)
            if "lo_c0.03" in r.index: ax.plot([r["lo_c0.03"], r["hi_c0.03"]], [y - 0.18, y - 0.18], color="tab:orange", lw=3, label="image-cluster units" if i == 0 else None)
            ax.plot([r.est], [y], "k|", ms=12)
        ax.axvline(0, color="k", lw=0.8); ax.set_title(f"split {sp}: ΔAUC with 95% intervals"); ax.set_yticks([-i for i in range(len(ids))]); ax.set_yticklabels([f"{c} {LABEL[c]}" for c in ids], fontsize=7)
    axes[0].legend(fontsize=7, loc="lower left"); fig1.tight_layout(); fig1.savefig(fig / "forest_auc.png", dpi=140); plt.close(fig1)
    if strata_conf is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True); sts = ["near", "mid", "far03", "far01"]
        for ax, sp in zip(axes, "AB"):
            for j, c in enumerate(["C3", "C4", "C5", "C2"]):
                for i, st in enumerate(sts):
                    r = row(strata_conf, c, sp, "auc", st)
                    if r is None: continue
                    x = i + (j - 1.5) * 0.18; ax.errorbar(x, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", color=f"C{j}", ms=4, label=f"{c} {LABEL[c]}" if i == 0 else None)
            ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(sts))); ax.set_xticklabels(["near (<=0.001)", "mid", "far (>0.03)", "far (>0.01)"]); ax.set_title(f"split {sp}: ΔAUC by anchor stratum (group 95% CI)")
        axes[0].legend(fontsize=6); fig2.tight_layout(); fig2.savefig(fig / "strata_auc.png", dpi=140); plt.close(fig2)
    try:
        key = pw[(pw.metric == "auc") & pw.comp.isin(["C3", "C4", "C5"])]; fig3, ax = plt.subplots(figsize=(5.5, 3.6))
        for (c, sp), g in key.groupby(["comp", "split"]): ax.bar(f"{c}/{sp}", g.mde80.iloc[0], color="tab:blue", alpha=0.6); ax.plot([f"{c}/{sp}"], [abs(g.est.iloc[0])], "ro")
        ax.set_ylabel("ΔAUC"); ax.set_title("bars: smallest gain detectable at 80% power (168 groups)\nred: observed gain", fontsize=8); fig3.tight_layout(); fig3.savefig(fig / "power_mde.png", dpi=140); plt.close(fig3)
    except Exception as e: print("power fig skipped", repr(e)[:100])
    (tab / "report_numbers.json").write_text(json.dumps(numbers, indent=1, default=str)); print("wrote", tab, fig)


if __name__ == "__main__": main()

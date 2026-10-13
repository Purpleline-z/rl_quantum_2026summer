"""Recomputes every pre-registered confirmation from the stored result JSONs, using each study's OWN test code (nd_confirm.py, gl_tables.py, sh_tables.py, gm_tables.py logic, ge_report.py) and its own
frozen spec files, and writes
  tables/confirmations.csv   one row per test (exp, block, family, split, metric, cells, n, gain, lo, hi, share_better, p, holm)
  tables/verdicts.csv        one row per pre-registered success rule (exp, claim, verdict, detail)
  tables/numbers_confirm.json  every gain/CI/p as a named number (keys  c.<exp>.<split>.<metric>.<cells-tag>.<field>)
Nothing is re-run; the numbers must equal the ones in the studies' own RESULTS documents (tests/test_crosscheck.py compares a sample)."""
from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import GE, ND, RES, Numbers

warnings.filterwarnings("ignore")
import nd_confirm as NC  # noqa: E402
import gl_tables as GL  # noqa: E402
import sh_tables as SH  # noqa: E402
import gm_tables as GM  # noqa: E402
import ge_analyze as GA  # noqa: E402
import ge_report as GR  # noqa: E402

N = Numbers("confirm"); TESTS = []; VERD = []


def tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", s).strip("_")


def spec_path(name):
    return GE / name if (GE / name).exists() else ND / name


def d(path): return [str(RES / p) for p in ([path] if isinstance(path, str) else path)]


def add(exp, block, out, kind="test"):
    """Normalise the output frame of one study's test code into the long table; register numbers."""
    out = out.copy()
    if "cells" not in out:
        def cellname(a, b):
            if a[0] == b[0]: return f"{a[1]} - {b[1]}"                                  # same selector (random): learner comparison
            if a[1] == b[1]: return f"{a[0]} - {b[0]} [{a[1]}]"                         # same learner: selection rule vs random
            return f"{a[0]}:{a[1]} - {b[0]}:{b[1]}"
        out["cells"] = [cellname(a, b) for a, b in zip(out.a, out.b)]
    for _, r in out.iterrows():
        row = dict(exp=exp, block=block, family=r["family"], split=r["split"], metric=r["metric"], cells=r["cells"], n=int(r["n_seeds"]), gain=float(r["gain"]), lo=float(r["lo"]), hi=float(r["hi"]),
                   share_better=float(r["share_better"]), p=float(r["p"]), holm=float(r["holm"]) if "holm" in r and not pd.isna(r["holm"]) else np.nan)
        TESTS.append(row)
        base = f"c.{exp}.{r['split']}.{r['metric']}.{tag(r['cells'])}"; key = base
        if r["family"].startswith(("secondary", "htr", "control")) or r["family"] in ("vs_aw8",): key = f"{base}.{tag(r['family'])}"
        elif exp.startswith("ND"): key = f"{base}.{tag(r['family'])}"
        for f_ in ("gain", "lo", "hi", "p", "holm", "n"):
            k = f"{key}.{f_}"
            if k not in N.d: N.put(k, row[f_ if f_ != "n" else "n"])


def verdict(exp, claim, ok, detail):
    VERD.append(dict(exp=exp, claim=claim, verdict=ok, detail=detail)); N.put(f"v.{exp}.{tag(claim)}", ok)


# ------------------------------------------------------------------------------------------------ new directions (specs + nd_confirm)
def run_nd():
    for exp, block, rel, specf in (("ND1", "3000-3034", "confirm1", "prereg1_spec.json"), ("ND3", "4000-4034", "confirm3_cold", "prereg3_spec.json"),
                                   ("ND4", "5000-5034", "confirm4_htr", "prereg4_spec.json"), ("ND5", "6000-6034", "confirm5_gp", "prereg5_spec.json")):
        spec = json.loads((ND / specf).read_text()); frame = NC.load(d(rel)); out = NC.run(spec, frame); add(exp, block, out)
        for fam in out.family.unique():
            o = out[out.family == fam]; pos = int((o.gain > 0).sum()); sig = int(((o.holm < 0.05) & (o.gain > 0)).sum())
            verdict(exp, f"family {fam}", "ALL SIGNIFICANT" if sig == len(o) else (f"{sig}/{len(o)} Holm<0.05 and positive"), f"{pos}/{len(o)} gains positive; {sig}/{len(o)} positive with Holm<0.05")


# ------------------------------------------------------------------------------------------------ graph learner (G1)
def run_gl():
    spec = json.loads((GE / "G1_spec.json").read_text())
    for exp, block, rel in (("GL1", "7100-7134", "gl/confirm1"), ("GL2", "7200-7234", "gl/confirm2"), ("GLC", "7300-7334 (cold start)", "gl/cold")):
        out, _ = GL.confirm_report(spec, d(rel)); add(exp, block, out); v = GL.verdicts(out, spec)
        for _, r in v.iterrows(): verdict(exp, f"{r.candidate} vs aw8_all", "CONFIRMED" if r.CONFIRMED else "NOT confirmed", f"all four positive={r.all_four_positive}; Holm-significant {r.n_holm_sig}/4; beats shuffled={r.beats_shuffled}")


# ------------------------------------------------------------------------------------------------ selection x learner x HTR (SH1/SH2)
def run_sh():
    sp_l = json.loads((GE / "SH1_spec_learner.json").read_text()); sp_c = json.loads((GE / "SH2_spec_cold.json").read_text()); sp_s = json.loads((GE / "SH1_spec_select.json").read_text()); sp_s4 = json.loads((GE / "SH1_spec_select_st4.json").read_text())
    jobs = [("SH1L", "8300-8334", "sh/c1_learn", sp_l, "learner"), ("SH2L", "8350-8384", "sh/c2_learn", sp_l, "learner"), ("SHCL", "8385-8399 (cold start)", "sh/cold_learn", sp_c, "learner"),
            ("SH1S", "8300-8334", "sh/c1_sel", sp_s, "select"), ("SH2S", "8350-8384", "sh/c2_sel", sp_s, "select"), ("SH1S4", "8300-8334 (learner lap3_k10_st4)", "sh/c1_sel4", sp_s4, "select")]
    for exp, block, rel, spec, kind in jobs:
        out, _ = SH.confirm_report(spec, d(rel)); add(exp, block, out)
        if kind == "learner":
            for _, r in SH.verdicts_learner(out, spec).iterrows():
                verdict(exp, f"lap3_k10_st4 vs {r.reference}", "CONFIRMED" if r.CONFIRMED else "NOT confirmed", f"all four positive={r.all_four_positive}; Holm-significant {r.n_holm_sig}/4; beats shuffled={r.beats_shuffled}; HTR acc Holm-sig splits={r.htr_acc_holm_sig_splits}, HTR ll Holm-sig splits={r.htr_ll_holm_sig_splits}; worst other-type acc gain={r.other_type_acc_worst_gain:+.4f}")
        else:
            for _, r in SH.verdicts(out, spec).iterrows():
                verdict(exp, f"selection {r.candidate}", "CONFIRMED" if r.MAIN_CONFIRMED else "NOT confirmed", f"all four positive={r.all_four_positive}; Holm-significant {r.n_holm_sig}/4; beats control={r.beats_control}; HTR acc Holm-sig splits={r.htr_acc_holm_sig_splits}, HTR ll Holm-sig splits={r.htr_ll_holm_sig_splits}; worst other-type acc gain={r.other_type_acc_worst_gain:+.4f}")


# ------------------------------------------------------------------------------------------------ graph mechanism (GM1): logic of gm_tables.py 'confirm' and 'dosetest' (their __main__ blocks are not importable)
def run_gm():
    spec = json.loads((GE / "GM1_spec.json").read_text()); ref, aw8, ng = spec["reference"], spec["aw8"], spec["nongraph"]
    for exp, block, rel, seeds in (("GM1", "8100-8134", "gm/confirm1", "8100-8134"), ("GM2", "8150-8184", "gm/confirm2", "8150-8184")):
        frame = GL.load(d(rel)); T = []; cell = lambda l: ["random", l]
        for cand, ctl in spec["candidates"].items():
            for split in spec["splits"]:
                for m in ("auc", "acc"):
                    T.append({"family": "main", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)}); T.append({"family": "vs_aw8", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(aw8)})
                    for k, name in ctl.items(): T.append({"family": f"control:{k}", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(name)})
                    T.append({"family": "control:nongraph", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ng)})
                for m in ("ll", "htr_acc", "htr_ll"): T.append({"family": f"secondary:{m}", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
        out = NC.run({"tests": T}, frame.assign(selector="random")); add(exp, block, out)
        for cand in spec["candidates"]:
            main = out[(out.family == "main") & (out.cand == cand)]; allpos = bool((main.gain > 0).all() and len(main) == 4); nsig = int((main.holm < 0.05).sum()); ok = {}
            for k in spec["candidates"][cand]:
                cc = out[(out.family == f"control:{k}") & (out.cand == cand)]; ok[k] = bool(((cc.gain > 0) & (cc.p < 0.05)).any())
            cc = out[(out.family == "control:nongraph") & (out.cand == cand)]; ok["nongraph"] = bool(((cc.gain > 0) & (cc.p < 0.05)).any())
            confirmed = bool(allpos and nsig >= 3 and all(v for k, v in ok.items() if k != "nongraph"))
            verdict(exp, f"{cand} vs lap3_k10", "CONFIRMED" if confirmed else "NOT confirmed", f"all four positive={allpos}; Holm-significant {nsig}/4; beats shuffled/noise controls={ {k: v for k, v in ok.items() if k != 'nongraph'} }; beats non-graph control={ok['nongraph']}")
        for sf, name in (("GM1_dose_primary.json", "dose-response (lambda=10; primary)"), ("GM1_dose_secondary.json", "dose-response (lambda=3; secondary, no verdict)")):
            ds = json.loads((GE / sf).read_text()); lo, hi = seeds.split("-"); seeds_ = set(range(int(lo), int(hi) + 1))   # the spec names block 1; the same test is applied to each block's own seeds
            dd, r = GM.dose_table(frame, seeds_); r = r[r["lambda"].isin(ds["lambdas"]) & r.metric.isin(ds["metrics"]) & r.split.isin(ds["splits"])].copy(); r["holm"] = NC.holm(r.p_wilcoxon.values); r["ps"] = (r.mean_spearman > 0) & (r.holm < 0.05)
            for _, x in r.iterrows():
                row = dict(exp=exp, block=block, family=f"dose:lambda{int(x['lambda'])}", split=x.split, metric=x.metric, cells="mean per-seed Spearman(gain over aw8_all, graph quality)", n=int(x.n_seeds), gain=float(x.mean_spearman), lo=np.nan, hi=np.nan, share_better=float(x.share_rho_positive), p=float(x.p_wilcoxon), holm=float(x.holm)); TESTS.append(row)
                for f_ in ("gain", "p", "holm", "n"): N.put(f"c.{exp}.dose{int(x['lambda'])}.{x.split}.{x.metric}.{f_}", row[f_])
            if ds["min_sig"] < 99: verdict(exp, name, "CONFIRMED" if int(r.ps.sum()) >= ds["min_sig"] else "NOT confirmed", f"{int(r.ps.sum())}/{len(r)} tests with positive Spearman and Holm<0.05 (need {ds['min_sig']})")


# ------------------------------------------------------------------------------------------------ graph endpoint (GE1): ge_report's own blocks and verdict functions + per-seed gains for intervals
def ge_pairs(frame, group_cols, arm_col, a, b, metrics=("auc", "acc")):
    per = frame.groupby(group_cols + ["seed", arm_col], as_index=False)[list(metrics)].mean(); res = {}
    for key, g in per.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics: piv = g.pivot(index="seed", columns=arm_col, values=m); res[key + (m,)] = (piv[a] - piv[b]).dropna()
    return res


def run_ge():
    for exp, blk, block in (("GE1", "conf1", "8500-8534"), ("GE2", "conf2", "8550-8584")):
        # (c) harsh splits
        _, resc = GR.harsh_block(blk, "t", bootstrap=False); f = GA.load(RES / "ge" / f"{blk}_harsh", "seed*_*.json"); gains = ge_pairs(f, ["kind"], "arm", "lap3", "base"); shuf = ge_pairs(f, ["kind"], "arm", "lap3", "lap3_shuf"); noisy = ge_pairs(f, ["kind"], "arm", "lap3", "lap3_n3")
        fam = [(k, v) for k, v in gains.items()]; ps = [L.wilcoxon_p(v) for _, v in fam]; hp = L.holm(ps)
        for ((kind, m), v), p, h in zip(fam, ps, hp):
            s = L.summarize(v); row = dict(exp=exp, block=block, family="harsh:H", split=f"harsh-{kind}", metric=m, cells="lap3_k10 - aw8_all", n=s["n"], gain=s["gain"], lo=s["lo"], hi=s["hi"], share_better=s["share_better"], p=p, holm=float(h)); TESTS.append(row)
            for f_ in ("gain", "lo", "hi", "p", "holm", "n"): N.put(f"c.{exp}.harsh-{kind}.{m}.lap3_vs_aw8_all.{f_}", row[f_])
        for lab, dd in (("shuf", shuf), ("noisy", noisy)):
            for (kind, m), v in dd.items():
                s = L.summarize(v); TESTS.append(dict(exp=exp, block=block, family=f"harsh:control:{lab}", split=f"harsh-{kind}", metric=m, cells=f"lap3_k10 - {lab} graph", n=s["n"], gain=s["gain"], lo=s["lo"], hi=s["hi"], share_better=s["share_better"], p=s["p"], holm=np.nan))
                for f_ in ("gain", "lo", "hi", "p"): N.put(f"c.{exp}.harsh-{kind}.{m}.lap3_vs_{lab}.{f_}", s[f_])
        verdict(exp, "harsh splits (c)", "MET" if GR.verdict_c(resc).endswith("MET") else "NOT met", GR.verdict_c(resc))
        # (b) dose-response trend (series 'all'); level gains
        _, resb = GR.dose_block(blk, "t"); f = GA.load(RES / "ge" / f"{blk}_dose", "seed*_*.json"); g = GA.dose_gains(f)
        for (split, var), gg in g.groupby(["split", "variant"]):
            for m in ("auc", "acc"):
                s = L.summarize(gg[m]); TESTS.append(dict(exp=exp, block=block, family="dose:level", split=split, metric=m, cells=f"lap3_k10 [{var}] - aw8_all", n=s["n"], gain=s["gain"], lo=s["lo"], hi=s["hi"], share_better=s["share_better"], p=s["p"], holm=np.nan))
                for f_ in ("gain", "lo", "hi", "p"): N.put(f"c.{exp}.dose.{split}.{m}.{var}.{f_}", s[f_])
        for _, t in resb["trend"].iterrows():
            TESTS.append(dict(exp=exp, block=block, family="dose:trend(all)", split=t.split, metric=t.metric, cells="slope of gain vs unlabelled fraction", n=int(t.n_seeds), gain=float(t.slope), lo=np.nan, hi=np.nan, share_better=float(t.share_pos), p=float(t.p), holm=float(t.holm)))
            for f_, v in (("gain", t.slope), ("p", t.p), ("holm", t.holm)): N.put(f"c.{exp}.dosetrend.{t.split}.{t.metric}.{f_}", float(v))
        verdict(exp, "unlabelled-image dose-response (b)", "MET" if GR.verdict_b(resb).endswith("MET") else "NOT met", GR.verdict_b(resb))
        # (a) classification
        _, resa = GR.class_block(blk, "t"); fc = GA.load(RES / "ge" / f"{blk}_class")
        for _, r in resa["k1"].iterrows():
            TESTS.append(dict(exp=exp, block=block, family="class:K1(vs 1-NN)", split="paper-split", metric="acc", cells=r["method"], n=int(r.n_seeds), gain=float(r["diff"]), lo=np.nan, hi=np.nan, share_better=float(r.better / max(r.n_seeds, 1)), p=float(r.p), holm=float(r.holm)))
        for _, r in resa["k1"].iterrows(): N.put(f"c.{exp}.class.{tag(r['method'])}.diff", float(r["diff"])); N.put(f"c.{exp}.class.{tag(r['method'])}.holm", float(r.holm))
        ra = GR.verdict_a(resa); verdict(exp, "classification beats 1-NN (a), inductive", "MET" if "inductive: MET" in ra else "NOT met", ra)


def main():
    run_nd(); run_gl(); run_sh(); run_gm(); run_ge()
    t = pd.DataFrame(TESTS); v = pd.DataFrame(VERD); t.to_csv(L.TABLES / "confirmations.csv", index=False); v.to_csv(L.TABLES / "verdicts.csv", index=False); N.save()
    L.write(L.TABLES / "verdicts.md", L.to_md(v, "{}")); print(len(t), "tests;", len(v), "verdicts")


if __name__ == "__main__":
    main()

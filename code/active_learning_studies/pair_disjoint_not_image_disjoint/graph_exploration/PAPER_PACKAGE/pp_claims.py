"""The claims table.  Every row = one claim; status, seeds, effect sizes with 95% intervals, the anchor-far check and the evidence pointer are generated from the stored results (tables/*.csv written by the other
pp_* scripts) -- only the claim wording, the choice of which cells belong to a claim, and the 'documented only' rows are written here by hand.

Status vocabulary (never anything stronger than what the rule/data say):
  PRE-REGISTERED, CONFIRMED          the frozen success rule was met in the primary block (+ 'replicated in ...' / 'NOT replicated in ...' for the further blocks)
  PRE-REGISTERED, NOT CONFIRMED      the frozen rule was not met
  PRE-REGISTERED RULE MET, CLAIM NOT SUPPORTED   (graph-endpoint dose-response: the formal trend test passes but the plateau check says 'more unlabelled images' does not help beyond 10 %)
  DESCRIPTIVE (confirmation seeds)   computed on pre-registered seeds but not itself a frozen test
  DEVELOPMENT OBSERVATION            development seeds only (selected on the same seeds: optimistic, winner's curse)
  AUDIT (red-team development)       red-team per-judgment audit on seeds 8800-8819, not a confirmation
  DOCUMENTED ONLY                    earlier protocol, numbers not regenerated in this package (appendix)
Outputs tables/claims_table.md (main table), tables/claims_appendix.md, tables/claims.csv, numbers_claims.json (keys cl.<id>.<field>)."""
import json
import re

import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import Numbers
from pp_registry import STUDIES

CONF = pd.read_csv(L.TABLES / "confirmations.csv"); VERD = pd.read_csv(L.TABLES / "verdicts.csv"); STR = pd.read_csv(L.TABLES / "strata.csv"); LAD = pd.read_csv(L.TABLES / "ladder.csv")
R = "random"; N = Numbers("claims")
BLOCK = {"ND1": "ND 3000-3034", "ND3": "ND 4000-4034 (cold start)", "ND4": "ND 5000-5034", "ND5": "ND 6000-6034", "GL1": "GL 7100-7134", "GL2": "GL 7200-7234", "GLC": "GL 7300-7334 (cold start)", "SH1L": "SH 8300-8334", "SH2L": "SH 8350-8384",
         "SHCL": "SH 8385-8399 (cold start, 15 seeds)", "SH1S": "SH 8300-8334", "SH2S": "SH 8350-8384", "SH1S4": "SH 8300-8334 (lap3_k10_st4)", "GM1": "GM 8100-8134", "GM2": "GM 8150-8184", "GE1": "GE 8500-8534", "GE2": "GE 8550-8584"}
EXP_STUDY = {"ND": "ND", "GL": "GL", "SH": "SH", "GM": "GM", "GE": "GE"}


def eff(exp, cells, fam=None, metrics=("auc", "acc"), splits=("A", "B")):
    x = CONF[(CONF.exp == exp) & (CONF.cells == cells)]
    if fam: x = x[x.family == fam]
    out = {}
    for s in splits:
        for m in metrics:
            r = x[(x.split == s) & (x.metric == m)]
            assert len(r) == 1, (exp, cells, fam, s, m, len(r)); out[(s, m)] = r.iloc[0]
    return out


def cell(e, m, splits=("A", "B"), d=3):
    return " / ".join(L.fmt_gain(e[(s, m)].gain, e[(s, m)].lo, e[(s, m)].hi, d) if not pd.isna(e[(s, m)].lo) else L.fmt_gain(e[(s, m)].gain, None, None, d) for s in splits)


def reg(cid, e, tagp=""):
    for (s, m), r in e.items():
        for f_ in ("gain", "lo", "hi", "p", "holm"):
            v = r[f_] if f_ in r else np.nan
            if not pd.isna(v): N.put(f"cl.{cid}{tagp}.{s}.{m}.{f_}", float(v))
        N.put(f"cl.{cid}{tagp}.{s}.{m}.n", int(r.n))


def dev_eff(dirs, a, b, selector=R, metrics=("auc", "acc"), a_sel=None):
    f = pd.concat([L.load_dir(d) for d in dirs], ignore_index=True); out = {}
    for s in ("A", "B"):
        for m in metrics:
            per = L.per_seed(f, s, m, (a_sel or selector, a), (selector, b)); r = L.summarize(per); out[(s, m)] = pd.Series(dict(gain=r["gain"], lo=r["lo"], hi=r["hi"], p=r["p"], n=r["n"], holm=np.nan))
    return out


def verdict_detail(exp, claim):
    v = VERD[(VERD.exp == exp) & (VERD.claim == claim)]; assert len(v) == 1, (exp, claim); return v.iloc[0].detail


def verdict_of(exp, claim):
    v = VERD[(VERD.exp == exp) & (VERD.claim == claim)]; assert len(v) == 1, (exp, claim); return v.iloc[0].verdict


def far_cell(cand, ref, stratum="far>0.03"):
    x = STR[(STR.source.str.startswith("dev")) & (STR.cand == cand) & (STR.ref == ref) & (STR.stratum == stratum) & (STR.metric == "auc")]
    if not len(x): return "not evaluated", "n/a"
    rows = {r.split: r for r in x.itertuples()}; txt = " / ".join(L.fmt_gain(rows[s].gain, rows[s].lo, rows[s].hi) for s in ("A", "B"))
    if all(r.lo > 0 for r in rows.values()): tag = "gain kept (CI>0 in A and B)"
    elif all(r.gain > 0 for r in rows.values()): tag = "point estimates >0, CI includes 0"
    elif all(r.hi < 0 for r in rows.values()): tag = "gain absent (negative)"
    else: tag = "not resolved / absent"
    return txt, tag


def near_cell(cand, ref):
    return far_cell(cand, ref, "near<=0.001")[0]


def evidence(exps, extra=""):
    parts = []
    for e in exps:
        st = STUDIES[re.match(r"[A-Z]+", e).group(0)[:2] if e[:2] in STUDIES else "ND"]; parts.append(f"{st['branch']}@{st['tip']}")
    return "; ".join(dict.fromkeys(parts)) + (" " + extra if extra else "")


ROWS = []
SHORT = {"L1": "aw8 (anchor weight 0.25 -> 8) beats the default head", "L2": "aw8_all (all ideal images as anchors) beats the default head", "L2b": "aw8_all vs aw8", "L3": "Laplacian kNN-graph regulariser (lap3_k10) beats aw8_all", "L4": "4x training steps (lap3_k10_st4) beats lap3_k10",
         "L5": "BEST: lap3_k10_st4 beats the tuned graph-free control", "L6": "whole recipe (lap3_k10_st4) vs aw8_all", "L7": "graph-free schedule tuning alone vs aw8_all", "L8": "real graph beats shuffled-graph control (a low bar)", "L9": "RT13-off variant beats lap3_k10 (not tested vs st4)",
         "L10": "uniform lambda 10 beats lap3_k10", "L11": "Laplacian + type-posterior term beats aw8_all", "L12": "GP learner beats the anchor-weighted MLP head", "L13": "aw8_all improves HTR vs default head", "D1": "long schedule does not help graph-free aw8_all",
         "D2": "graph-free consistency regulariser comes close (cons - st4 shown)", "G1": "gain tracks graph quality (noise dose-response; rho)", "G2": "10% random edge rewiring destroys the gain; local rewiring harmless", "G3": "shuffled graph is far worse than no graph (lambda 10)",
         "G4": "more unlabelled frames: gain saturates at ~10% of frames", "G5": "harsh splits (held-out run / temperature; fixed test sets): gain kept", "G6": "paper's type-accuracy endpoint: no graph advantage", "S1": "vopt_u beats random with the default head",
         "S2": "vopt_u beats random with the aw8 head", "S3": "4 selection rules beat random on the lap3_k10 learner", "S4": "HTR-directed selection (vopt_htr4)", "S5": "HTR: st4 vs tuned graph-free control", "S6": "HTR: st4 vs lap3_k10",
         "R1": "gain concentrated in anchor-near judgments", "R2": "aw8 gain is not an anchor-copy artefact", "R3": "time-blocked split keeps the gain"}


def add(cid, topic, text, status, seeds, auc, acc, far=("", ""), repl="", evid="", kind=""):
    ROWS.append(dict(id=cid, topic=topic, claim=text, status=status, seeds=seeds, dAUC=auc, dACC=acc, anchor_far=far[0], anchor_far_status=far[1], replications=repl, evidence=evid, kind=kind))
    N.put(f"cl.{cid}.status", status)


def primary_and_repl(exps, claim_names):
    """status from the frozen success rule (verdict of the first block) + replication blocks"""
    v = [verdict_of(e, c) for e, c in zip(exps, claim_names)]; ok = lambda s: s in ("CONFIRMED", "MET", "ALL SIGNIFICANT")

    def partial(s):
        m = re.match(r"(\d+)/(\d+) Holm", s); return (int(m.group(1)), int(m.group(2))) if m and int(m.group(1)) > 0 else None
    if not ok(v[0]): return "PRE-REGISTERED, NOT CONFIRMED" + (f" (other blocks: {', '.join(e + ': ' + x for e, x in zip(exps[1:], v[1:]))})" if len(exps) > 1 else "")
    st = "PRE-REGISTERED, CONFIRMED"
    if len(exps) > 1:
        yes = [e for e, x in zip(exps[1:], v[1:]) if ok(x)]; part = [f"{e} ({partial(x)[0]}/{partial(x)[1]} tests Holm<0.05)" for e, x in zip(exps[1:], v[1:]) if not ok(x) and partial(x)]; no = [e for e, x in zip(exps[1:], v[1:]) if not ok(x) and not partial(x)]
        st += (f"; replicated in {', '.join(yes)}" if yes else "") + (f"; PARTLY replicated in {', '.join(part)}" if part else "") + (f"; NOT replicated in {', '.join(no)} -> not robust" if no else "")
    return st


def htr_counts(exp, claim):
    d = verdict_detail(exp, claim); return int(re.search(r"HTR acc Holm-sig splits=(\d)", d).group(1)), int(re.search(r"HTR ll Holm-sig splits=(\d)", d).group(1))


def repl_text(items, m="auc"):
    out = []
    for exp, cells, fam in items:
        e = eff(exp, cells, fam, metrics=(m,)); out.append(f"{exp}: {L.fmt_gain(e[('A', m)].gain)} / {L.fmt_gain(e[('B', m)].gain)}")
    return "; ".join(out)


def build():
    # ------------------------------------------------------------------------------------------------ learner ladder
    e = eff("ND1", "aw8 - baseline", "F1"); reg("L1", e)
    add("L1", "learner", "Anchor weight 0.25 -> 8 (aw8) improves held-out preference prediction over the default head", primary_and_repl(["ND1", "ND3"], ["family F1", "family G1"]), BLOCK["ND1"] + " (35 seeds); replication " + BLOCK["ND3"],
        cell(e, "auc"), cell(e, "acc"), far_cell("aw8", "baseline"), "ND3 AUC A/B " + L.fmt_gain(eff("ND3", "aw8 - baseline", "G1")[("A", "auc")].gain) + " / " + L.fmt_gain(eff("ND3", "aw8 - baseline", "G1")[("B", "auc")].gain), evidence(["ND1", "ND3"], "(PREREGISTRATION_1/3)"))
    e = eff("ND1", "aw8_all - baseline", "F1"); reg("L2", e)
    add("L2", "learner", "All 144 ideal images as anchors (aw8_all) improve over the default head", primary_and_repl(["ND1"], ["family F1"]), BLOCK["ND1"], cell(e, "auc"), cell(e, "acc"), far_cell("aw8_all", "baseline"), "", evidence(["ND1"], "(PREREGISTRATION_1)"))
    la = LAD[(LAD.step == "L2") & (LAD.block == "ND4") & LAD.metric.isin(["auc", "acc"])]; e2 = {(r.split, r.metric): r for r in la.itertuples()}
    fmt2 = lambda m: " / ".join(L.fmt_gain(e2[(s, m)].gain, e2[(s, m)].lo, e2[(s, m)].hi) for s in ("A", "B"))
    for (s, m), r in e2.items(): N.put(f"cl.L2b.{s}.{m}.gain", float(r.gain)); N.put(f"cl.L2b.{s}.{m}.lo", float(r.lo)); N.put(f"cl.L2b.{s}.{m}.hi", float(r.hi))
    add("L2b", "learner", "aw8_all vs aw8 (the extra step of using all ideal images)", "DESCRIPTIVE (confirmation seeds): difference of two pre-registered contrasts, not itself a frozen test", BLOCK["ND4"], fmt2("auc"), fmt2("acc"), far_cell("aw8_all", "aw8"), "", evidence(["ND4"]))
    e = eff("GL1", "lap3_k10 - aw8_all", "main"); reg("L3", e)
    add("L3", "learner", "Laplacian regulariser on the cosine-kNN graph of unlabelled images (lap3_k10) improves over aw8_all", primary_and_repl(["GL1", "GL2", "GLC"], ["lap3_k10 vs aw8_all"] * 3), BLOCK["GL1"] + "; replication " + BLOCK["GL2"] + "; cold start " + BLOCK["GLC"], cell(e, "auc"), cell(e, "acc"), far_cell("lap3_k10", "aw8_all"),
        repl_text([("GL2", "lap3_k10 - aw8_all", "main"), ("GLC", "lap3_k10 - aw8_all", "main")]), evidence(["GL1"], "(PREREGISTRATION_G1)"))
    e = eff("SH1L", "lap3_k10_st4 - lap3_k10", "main:lap3_k10"); reg("L4", e)
    add("L4", "learner", "Four times the head's training steps on top of the Laplacian learner (lap3_k10_st4 vs lap3_k10; secondary family)", primary_and_repl(["SH1L", "SH2L", "SHCL"], ["lap3_k10_st4 vs lap3_k10"] * 3), BLOCK["SH1L"] + "; replication " + BLOCK["SH2L"] + "; cold start " + BLOCK["SHCL"], cell(e, "auc"), cell(e, "acc"), far_cell("lap3_k10_st4", "lap3_k10"),
        repl_text([("SH2L", "lap3_k10_st4 - lap3_k10", "main:lap3_k10"), ("SHCL", "lap3_k10_st4 - lap3_k10", "main:lap3_k10")]), evidence(["SH1L"], "(PREREGISTRATION_SH1/SH2)"))
    e = eff("SH1L", "lap3_k10_st4 - aw8_all_st2lr0.5", "main:aw8_all_st2lr0.5"); reg("L5", e)
    add("L5", "learner", "CURRENT BEST: lap3_k10_st4 beats the tuned graph-free control aw8_all_st2lr0.5 (primary pre-registered test)", primary_and_repl(["SH1L", "SH2L", "SHCL"], ["lap3_k10_st4 vs aw8_all_st2lr0.5"] * 3), BLOCK["SH1L"] + "; replication " + BLOCK["SH2L"] + "; cold start " + BLOCK["SHCL"], cell(e, "auc"), cell(e, "acc"), far_cell("lap3_k10_st4", "aw8_all_st2lr0.5"),
        repl_text([("SH2L", "lap3_k10_st4 - aw8_all_st2lr0.5", "main:aw8_all_st2lr0.5"), ("SHCL", "lap3_k10_st4 - aw8_all_st2lr0.5", "main:aw8_all_st2lr0.5")]), evidence(["SH1L"], "(PREREGISTRATION_SH1/SH2)"))
    for cid, step, text, a, b in (("L6", "T", "Whole recipe: lap3_k10_st4 vs aw8_all with the repository schedule", "lap3_k10_st4", "aw8_all"), ("L7", "C1", "Graph-free schedule tuning alone (aw8_all_st2lr0.5 vs aw8_all)", "aw8_all_st2lr0.5", "aw8_all")):
        la = LAD[(LAD.step == step) & (LAD.block == "SH1") & LAD.metric.isin(["auc", "acc"])]; e2 = {(r.split, r.metric): r for r in la.itertuples()}; fm = lambda m: " / ".join(L.fmt_gain(e2[(s, m)].gain, e2[(s, m)].lo, e2[(s, m)].hi) for s in ("A", "B"))
        for (s, m), r in e2.items(): N.put(f"cl.{cid}.{s}.{m}.gain", float(r.gain)); N.put(f"cl.{cid}.{s}.{m}.lo", float(r.lo)); N.put(f"cl.{cid}.{s}.{m}.hi", float(r.hi))
        add(cid, "learner", text, "DESCRIPTIVE (confirmation seeds): not a frozen test", BLOCK["SH1L"], fm("auc"), fm("acc"), far_cell(a, b), "", evidence(["SH1L"]))
    la = LAD[(LAD.step == "C3") & (LAD.block == "SH1") & LAD.metric.isin(["auc", "acc"])]; e2 = {(r.split, r.metric): r for r in la.itertuples()}; fm = lambda m: " / ".join(L.fmt_gain(e2[(s, m)].gain, e2[(s, m)].lo, e2[(s, m)].hi) for s in ("A", "B"))
    for (s, m), r in e2.items(): N.put(f"cl.L8.{s}.{m}.gain", float(r.gain)); N.put(f"cl.L8.{s}.{m}.lo", float(r.lo)); N.put(f"cl.L8.{s}.{m}.hi", float(r.hi))
    add("L8", "learner", "Real graph beats its shuffled-graph control by a large margin (a LOW bar: the shuffled graph is far worse than no graph at all, see G3)", "PRE-REGISTERED CONTROL (uncorrected): real graph better", BLOCK["SH1L"], fm("auc"), fm("acc"), ("not evaluated", "n/a"), "", evidence(["SH1L"]))
    e = eff("GM1", "gmc_t_no_rt13 - lap3_k10", "main"); reg("L9", e)
    add("L9", "learner", "RT13 scores excluded from the graph penalty (gmc_t_no_rt13, lambda 10) beats lap3_k10; NOT compared with lap3_k10_st4", primary_and_repl(["GM1", "GM2"], ["gmc_t_no_rt13 vs lap3_k10"] * 2), BLOCK["GM1"] + "; replication " + BLOCK["GM2"], cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"),
        repl_text([("GM2", "gmc_t_no_rt13 - lap3_k10", "main")]), evidence(["GM1"], "(PREREGISTRATION_GM1)"))
    e = eff("GM1", "gm_lap10_n0 - lap3_k10", "main"); reg("L10", e)
    add("L10", "learner", "Uniform lambda 10 (gm_lap10_n0) beats lap3_k10", primary_and_repl(["GM1", "GM2"], ["gm_lap10_n0 vs lap3_k10"] * 2) + " -> NOT robustly confirmed", BLOCK["GM1"] + "; " + BLOCK["GM2"], cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), repl_text([("GM2", "gm_lap10_n0 - lap3_k10", "main")]), evidence(["GM1"], "(PREREGISTRATION_GM1)"))
    e = eff("GL1", "lap3_rq3_k10 - aw8_all", "main"); reg("L11", e)
    add("L11", "learner", "Laplacian + type-posterior ranking term (lap3_rq3_k10) beats aw8_all", primary_and_repl(["GL1", "GL2", "GLC"], ["lap3_rq3_k10 vs aw8_all"] * 3), BLOCK["GL1"] + "; " + BLOCK["GL2"] + "; " + BLOCK["GLC"], cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), repl_text([("GL2", "lap3_rq3_k10 - aw8_all", "main")]), evidence(["GL1"], "(PREREGISTRATION_G1)"))
    e = eff("ND5", "gp_aw8 - aw8", "K1"); reg("L12", e)
    add("L12", "learner", "GP preference learner (with aw8 anchors) beats the anchor-weighted MLP head (rule: AUC and accuracy both significant)", "PRE-REGISTERED, NOT CONFIRMED (AUC gain significant, accuracy not)", BLOCK["ND5"], cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), "", evidence(["ND5"], "(PREREGISTRATION_5)"))
    e = eff("ND4", "aw8_all - baseline", "H1", metrics=("htr_acc", "htr_ll")); reg("L13", e)
    add("L13", "HTR", "aw8_all improves HTR accuracy / log-loss over the default head (HTR = lab's priority type)", "PRE-REGISTERED, NOT CONFIRMED (0 of 12 HTR tests Holm-significant)", BLOCK["ND4"], "HTR acc: " + cell(e, "htr_acc"), "HTR ll: " + cell(e, "htr_ll"), ("not evaluated", "n/a"), "", evidence(["ND4"], "(PREREGISTRATION_4)"), kind="htr")
    # ------------------------------------------------------------------------------------------------ development observations on the learner
    e = dev_eff(["sh/sdev5", "sh/sdev1"], "aw8_all_st4", "aw8_all"); reg("D1", e)
    add("D1", "learner", "Longer head schedule (4x steps) does NOT help the graph-free aw8_all (it needs the regulariser to tame overfitting)", "DEVELOPMENT OBSERVATION", "SH dev 8200-8219 (20 seeds)", cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), "", evidence(["SH1L"], "(sdev5, sdev1)"))
    e = dev_eff(["sh/sdev9", "sh/sdev5"], "cons0.3_st4", "lap3_k10_st4"); reg("D2", e)
    add("D2", "mechanism", "A graph-free consistency regulariser at the long schedule (cons0.3_st4) comes close to lap3_k10_st4 (difference shown = cons - lap3_k10_st4): the graph's own share is small and unconfirmed", "DEVELOPMENT OBSERVATION", "SH dev 8200-8219 (20 seeds)", cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), "", evidence(["SH1L"], "(sdev9, sdev5)"))
    # ------------------------------------------------------------------------------------------------ graph mechanism / endpoint
    dn = pd.read_csv(L.TABLES / "dose_noise.csv"); dr = pd.read_csv(L.TABLES / "dose_rewire.csv"); dg = pd.read_csv(L.TABLES / "dose_nodes.csv")
    x = CONF[(CONF.exp == "GM1") & (CONF.family == "dose:lambda10")]; sp = {(r.split, r.metric): r for r in x.itertuples()}
    cellsp = lambda m: " / ".join(f"rho {sp[(s, m)].gain:+.2f}" for s in ("A", "B"))
    for (s, m), r in sp.items(): N.put(f"cl.G1.{s}.{m}.rho", float(r.gain)); N.put(f"cl.G1.{s}.{m}.holm", float(r.holm))
    xx = CONF[(CONF.exp == "GM2") & (CONF.family == "dose:lambda10")]
    for r in xx.itertuples(): N.put(f"cl.G1r.{r.split}.{r.metric}.rho", float(r.gain)); N.put(f"cl.G1r.{r.split}.{r.metric}.holm", float(r.holm))
    add("G1", "graph", "Along the noise axis the Laplacian gain tracks graph quality (per-seed Spearman between gain and ideal-ideal same-class share; lambda 10; effect columns show mean rho)", primary_and_repl(["GM1", "GM2"], ["dose-response (lambda=10; primary)"] * 2), BLOCK["GM1"] + "; replication " + BLOCK["GM2"], "AUC " + cellsp("auc"), "acc " + cellsp("acc"), ("not evaluated", "n/a"), "GM2 rho(AUC) A/B " + " / ".join(f"{float(xx[(xx.split == s) & (xx.metric == 'auc')].gain.iloc[0]):+.2f}" for s in ("A", "B")), evidence(["GM1"], "(PREREGISTRATION_GM1)"))
    d_ = lambda k, fr, s, m, f_: float(dr[(dr.kind == k) & (dr.frac == fr) & (dr.split == s) & (dr.metric == m)][f_].iloc[0])
    e = {(s, "auc"): pd.Series(dict(gain=d_("random", 10, s, "auc", "gain"), lo=d_("random", 10, s, "auc", "lo"), hi=d_("random", 10, s, "auc", "hi"), n=20, p=np.nan, holm=np.nan)) for s in ("A", "B")}; e.update({(s, "acc"): pd.Series(dict(gain=d_("random", 10, s, "acc", "gain"), lo=d_("random", 10, s, "acc", "lo"), hi=d_("random", 10, s, "acc", "hi"), n=20, p=np.nan, holm=np.nan)) for s in ("A", "B")}); reg("G2", e)
    add("G2", "graph", "Randomly rewiring only 10 % of the graph's edges turns the lambda-10 gain over aw8_all into a loss; local rewiring (neighbour -> another of its 50 nearest) is harmless", "DEVELOPMENT OBSERVATION", "GM dev 8000-8019 (20 seeds)", cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"),
        "local 50 %: AUC A/B " + " / ".join(L.fmt_gain(d_("local", 50, s, "auc", "gain")) for s in ("A", "B")), evidence(["GM1"], "(dev16, dev17)"))
    d2 = lambda b, k, s, m, f_: float(dn[(dn.block == b) & (dn.learner == k) & (dn.split == s) & (dn.metric == m)][f_].iloc[0])
    e = {(s, m): pd.Series(dict(gain=d2("GM1", "gm_lap10_n0_shuf", s, m, "gain"), lo=d2("GM1", "gm_lap10_n0_shuf", s, m, "lo"), hi=d2("GM1", "gm_lap10_n0_shuf", s, m, "hi"), n=35, p=np.nan, holm=np.nan)) for s in ("A", "B") for m in ("auc", "acc")}; reg("G3", e)
    add("G3", "graph", "A shuffled graph (node identities permuted, lambda 10) is far WORSE than no graph at all (gain over aw8_all shown); i.e. 'beats the shuffled control' is a low bar", "PRE-REGISTERED CONTROL (uncorrected)", BLOCK["GM1"], cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), "", evidence(["GM1"]))
    g_ = lambda b, v, s, m, f_: float(dg[(dg.block == b) & (dg.variant == v) & (dg.split == s) & (dg.metric == m)][f_].iloc[0])
    e = {(s, m): pd.Series(dict(gain=g_("GE1", "all_p100", s, m, "gain"), lo=g_("GE1", "all_p100", s, m, "lo"), hi=g_("GE1", "all_p100", s, m, "hi"), n=35, p=np.nan, holm=np.nan)) for s in ("A", "B") for m in ("auc", "acc")}; reg("G4", e)
    for v_ in ("all_p0", "all_p10", "all_p25", "all_p50"):
        for s in ("A", "B"): N.put(f"cl.G4.{v_}.{s}.auc.gain", g_("GE1", v_, s, "auc", "gain"))
    add("G4", "graph", "More unlabelled images in the graph: the formal trend test passes, but the whole trend is the step from 0 to 10 % of the trajectory frames; nothing consistent from 10 % to 100 % (effect shown = lap3_k10 over aw8_all with 100 % of the frames)", "PRE-REGISTERED RULE MET (GE1 and GE2), PRACTICAL CLAIM 'more unlabelled images help' NOT SUPPORTED (plateau)", BLOCK["GE1"] + "; " + BLOCK["GE2"],
        cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), "gain at 0 / 10 % / 100 % of frames, AUC A: " + " / ".join(L.fmt_gain(g_("GE1", v_, "A", "auc", "gain")) for v_ in ("all_p0", "all_p10", "all_p100")), evidence(["GE1"], "(PREREGISTRATION_GE1)"))
    e = {(s, m): CONF[(CONF.exp == "GE1") & (CONF.family == "harsh:H") & (CONF.split == s) & (CONF.metric == m)].iloc[0] for s in ("harsh-run", "harsh-temp") for m in ("auc", "acc")}; reg("G5", e)
    bc = json.loads((L.RES / "ge" / "bootstrap_cache.json").read_text()); get = lambda k: next(v for kk, v in bc.items() if kk.startswith(k))
    for blk in ("conf1", "conf2"):
        for kind in ("run", "temp"):
            for m in ("auc", "acc"): g = get(f"{blk}_harsh|{kind}|lap3|base"); N.put(f"cl.G5.tgb.{blk}.{kind}.{m}.gain", g[m][0]); N.put(f"cl.G5.tgb.{blk}.{kind}.{m}.lo", g[m][1]); N.put(f"cl.G5.tgb.{blk}.{kind}.{m}.hi", g[m][2])
    n_incl = sum(1 for blk in ("conf1", "conf2") for kind in ("run", "temp") for m in ("auc", "acc") if get(f"{blk}_harsh|{kind}|lap3|base")[m][1] <= 0 <= get(f"{blk}_harsh|{kind}|lap3|base")[m][2]); N.put("cl.G5.tgb.n_ci_include_zero_of_8", n_incl)
    add("G5", "graph", "Harsh splits: lap3_k10 keeps its gain over aw8_all when a whole growth run / the top temperature tercile is held out (columns = held-out run / held-out temperature; FIXED test sets of 50 / 26 judgments, so seed-level p-values reflect training variability only; test-group bootstrap CIs include 0 in " + str(n_incl) + " of 8)",
        primary_and_repl(["GE1", "GE2"], ["harsh splits (c)"] * 2) + " (GE2 uses the same test judgments: not independent)", BLOCK["GE1"] + "; " + BLOCK["GE2"], cell(e, "auc", splits=("harsh-run", "harsh-temp")), cell(e, "acc", splits=("harsh-run", "harsh-temp")), ("not evaluated", "n/a"), "", evidence(["GE1"], "(PREREGISTRATION_GE1)"))
    cl = CONF[(CONF.exp.isin(["GE1", "GE2"])) & (CONF.family == "class:K1(vs 1-NN)") & (CONF.cells == "laplr (inductive)")]; c1, c2 = cl[cl.exp == "GE1"].iloc[0], cl[cl.exp == "GE2"].iloc[0]
    for nm, r in (("GE1", c1), ("GE2", c2)): N.put(f"cl.G6.{nm}.laplr_ind.diff", float(r.gain)); N.put(f"cl.G6.{nm}.laplr_ind.holm", float(r.holm))
    for nm in ("GE1", "GE2"):
        r = CONF[(CONF.exp == nm) & (CONF.family == "class:K1(vs 1-NN)") & (CONF.cells.str.startswith("lap3_ref"))].iloc[0]; N.put(f"cl.G6.{nm}.lap3_ref.diff", float(r.gain)); N.put(f"cl.G6.{nm}.lap3_ref.holm", float(r.holm))
    cn = CONF[(CONF.family == "class:control")]; lg = {r.exp: r for r in cn[cn.cells == "logreg_vs_nn"].itertuples()}; ll_ = {r.exp: r for r in cn[cn.cells == "laplr_vs_logreg"].itertuples()}
    for e_ in ("GE1", "GE2"): N.put(f"cl.G6.{e_}.logreg_vs_nn", float(lg[e_].gain)); N.put(f"cl.G6.{e_}.laplr_vs_logreg", float(ll_[e_].gain))
    add("G6", "graph", f"Paper's own endpoint (type accuracy of 28 outer-test ideal images): no graph method beats 1-NN in a way attributable to the graph (effect = laplr inductive minus 1-NN, accuracy; plain logistic regression minus 1-NN: GE1 {lg['GE1'].gain:+.3f}, GE2 {lg['GE2'].gain:+.3f}; laplr minus logistic regression: GE1 {ll_['GE1'].gain:+.3f}, GE2 {ll_['GE2'].gain:+.3f}; the Laplacian reward head is significantly WORSE than 1-NN)", "PRE-REGISTERED RULE: NOT MET in GE1, MET in GE2 (by laplr; most of its gain is plain logistic regression's own gain over 1-NN, laplr minus logistic regression is small: see the cell text)", BLOCK["GE1"] + "; " + BLOCK["GE2"],
        "n/a (accuracy endpoint)", f"GE1 {c1.gain:+.3f} (Holm {c1.holm:.2f}); GE2 {c2.gain:+.3f} (Holm {c2.holm:.3f})", ("not evaluated", "n/a"), "", evidence(["GE1"], "(PREREGISTRATION_GE1)"))
    # ------------------------------------------------------------------------------------------------ selection
    e = eff("ND1", "vopt_u - random [baseline]", "F3"); reg("S1", e)
    add("S1", "selection", "With the default head the pool-wide I-optimal rule vopt_u beats random selection", primary_and_repl(["ND1", "ND3"], ["family F3", "family G3"]), BLOCK["ND1"] + "; " + BLOCK["ND3"], cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), repl_text([("ND3", "vopt_u - random [baseline]", "G3")]), evidence(["ND1"], "(PREREGISTRATION_1)"))
    e = eff("ND1", "vopt_u - random [aw8]", "F2"); reg("S2", e)
    add("S2", "selection", "With the anchor-weighted head (aw8) vopt_u no longer beats random selection: the learner change absorbs the selection gain", primary_and_repl(["ND1"], ["family F2"]) , BLOCK["ND1"] + "; cold start " + BLOCK["ND3"] + " also not confirmed", cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), repl_text([("ND3", "vopt_u - random [aw8]", "G2")]), evidence(["ND1"], "(PREREGISTRATION_1)"))
    e = eff("SH1S", "vopt_u - random", "main"); reg("S3", e)
    nsig = [verdict_of(x, f"selection {c}") for x in ("SH1S", "SH2S", "SH1S4") for c in ("vopt_u", "gvopt_node3", "typed_decisive_bald", "vopt_htr4")]
    add("S3", "selection", "On top of the Laplacian learner none of four pre-registered selection rules (vopt_u, graph-aware gvopt_node3, typed_decisive_bald, vopt_htr4) beats random on AUC/accuracy (shown: vopt_u)", "PRE-REGISTERED, NOT CONFIRMED (" + str(sum(v == 'NOT confirmed' for v in nsig)) + " of " + str(len(nsig)) + " rule x block verdicts: not confirmed; blocks SH1, SH2, and SH1 under lap3_k10_st4)", BLOCK["SH1S"] + "; " + BLOCK["SH2S"] + "; " + BLOCK["SH1S4"],
        cell(e, "auc"), cell(e, "acc"), ("not evaluated", "n/a"), repl_text([("SH2S", "vopt_u - random", "main"), ("SH1S4", "vopt_u - random", "main")]), evidence(["SH1S"], "(PREREGISTRATION_SH1)"))
    e = eff("SH1S", "vopt_htr4 - random", "htr", metrics=("htr_acc", "htr_ll")); reg("S4", e); e2_ = eff("SH2S", "vopt_htr4 - random", "htr", metrics=("htr_acc", "htr_ll")); reg("S4r", e2_)
    cnt = [htr_counts(x, "selection vopt_htr4") for x in ("SH1S", "SH2S", "SH1S4")]
    add("S4", "HTR", "HTR-directed selection (vopt_htr4, 43 % of picks on HTR vs 25 % random) improves HTR log-loss, but HTR accuracy only marginally and overall AUC/accuracy not at all", "PRE-REGISTERED HTR FAMILY: HTR log-loss Holm-significant in " + ", ".join(f"{c[1]}/2" for c in cnt) + " splits (SH1S, SH2S, SH1S4), HTR accuracy in " + ", ".join(f"{c[0]}/2" for c in cnt) + " (rule needs both metrics in both splits): HTR IMPROVEMENT NOT CONFIRMED on accuracy; log-loss gain real but small",
        BLOCK["SH1S"] + "; " + BLOCK["SH2S"], "HTR acc: " + cell(e, "htr_acc"), "HTR ll: " + cell(e, "htr_ll"), ("not evaluated", "n/a"), "SH2S HTR ll A/B " + " / ".join(L.fmt_gain(e2_[(s, "htr_ll")].gain) for s in ("A", "B")), evidence(["SH1S"], "(PREREGISTRATION_SH1)"), kind="htr")
    for cid, ref, txt in (("S5", "aw8_all_st2lr0.5", "HTR: lap3_k10_st4 improves HTR accuracy and log-loss over the tuned graph-free control"), ("S6", "lap3_k10", "HTR: lap3_k10_st4 improves HTR accuracy and log-loss over lap3_k10")):
        ex = [("SH1L", "SH1L"), ("SH2L", "SH2L"), ("SHCL", "SHCL")]; cn = [htr_counts(x, f"lap3_k10_st4 vs {ref}") for x, _ in ex]; okb = [c[0] == 2 and c[1] == 2 for c in cn]
        e = eff("SH1L", f"lap3_k10_st4 - {ref}", f"htr:{ref}", metrics=("htr_acc", "htr_ll")); reg(cid, e)
        add(cid, "HTR", txt, ("PRE-REGISTERED HTR RULE MET in " + ", ".join(x for (x, _), o in zip(ex, okb) if o) if any(okb) else "PRE-REGISTERED HTR RULE NOT MET") + ("; NOT met in " + ", ".join(x for (x, _), o in zip(ex, okb) if not o) + " -> not robust" if not all(okb) else " (all blocks)") + " (Holm-significant splits for accuracy/log-loss: " + ", ".join(f"{x} {c[0]}/{c[1]}" for (x, _), c in zip(ex, cn)) + ")",
            BLOCK["SH1L"] + "; " + BLOCK["SH2L"] + "; " + BLOCK["SHCL"], "HTR acc: " + cell(e, "htr_acc"), "HTR ll: " + cell(e, "htr_ll"), ("not evaluated", "n/a"), "HTR acc A/B: " + repl_text([(x, f"lap3_k10_st4 - {ref}", f"htr:{ref}") for x in ("SH2L", "SHCL")], "htr_acc"), evidence(["SH1L"], "(PREREGISTRATION_SH1/SH2)"), kind="htr")

    # ------------------------------------------------------------------------------------------------ red team / limits
    far, tag = far_cell("lap3_k10_st4", "aw8_all_st2lr0.5"); near = near_cell("lap3_k10_st4", "aw8_all_st2lr0.5")
    for s in ("A", "B"):
        for st in ("near<=0.001", "far>0.03", "far>0.01 (red-team threshold)", "all"):
            r = STR[(STR.source.str.startswith("dev")) & (STR.cand == "lap3_k10_st4") & (STR.ref == "aw8_all_st2lr0.5") & (STR.split == s) & (STR.stratum == st) & (STR.metric == "auc")].iloc[0]
            for f_ in ("gain", "lo", "hi"): N.put(f"cl.R1.{s}.{st}.{f_}", float(r[f_]))
    allc = lambda cand, ref: " / ".join(L.fmt_gain(*(STR[(STR.source.str.startswith("dev")) & (STR.cand == cand) & (STR.ref == ref) & (STR.split == s) & (STR.stratum == "all") & (STR.metric == "auc")][["gain", "lo", "hi"]].iloc[0])) for s in ("A", "B"))
    add("R1", "limits", "The learner gain over the graph-free control is concentrated in judgments with an image that is a (near-)copy of a typed anchor (accuracy column shows the anchor-NEAR stratum <= 0.001, ΔAUC; the AUC column is the all-judgments ΔAUC of these 20 seeds)", "AUDIT (red-team development, 20 seeds): not a confirmation", "RT dev 8800-8819 (20 seeds); RT1 confirmation block 8850-8884 used automatically if its files exist (see strata.csv)", allc("lap3_k10_st4", "aw8_all_st2lr0.5"), "near<=0.001: " + near, (far, tag), "", evidence(["RT"], "(rt/a1, a1b)"), kind="audit")
    far2, tag2 = far_cell("aw8_all", "baseline")
    add("R2", "limits", "The anchor-weight gain (aw8_all vs default head) is NOT an anchor-copy artefact: it is as large or larger in anchor-far judgments (accuracy column shows the anchor-NEAR stratum ΔAUC)", "AUDIT (red-team development, 20 seeds): not a confirmation", "RT dev 8800-8819, budgets 20/40, 2 draws", allc("aw8_all", "baseline"), "near<=0.001: " + near_cell("aw8_all", "baseline"), (far2, tag2), "", evidence(["RT"], "(rt/a1c, a1b)"), kind="audit")
    rx = pd.read_csv(L.TABLES / "redteam_extra.csv") if (L.TABLES / "redteam_extra.csv").exists() and (L.TABLES / "redteam_extra.csv").stat().st_size > 5 else pd.DataFrame()
    if len(rx[rx.audit == "a3"]) if len(rx) else 0:
        g = lambda ref, m, f_: float(rx[(rx.audit == "a3") & (rx.cand == "lap3_k10_st4") & (rx.ref == ref) & (rx.group == "blocked") & (rx.metric == m)][f_].iloc[0])
        for ref_, tag_ in (("aw8_all_st2lr0.5", ""), ("lap3_k10", "b"), ("lap3_k10_st4_shuf", "c")):
            for m in ("auc", "acc"):
                for f_ in ("gain", "lo", "hi"): N.put(f"cl.R3{tag_}.{m}.{f_}", g(ref_, m, f_))
        fm = lambda m: L.fmt_gain(g("aw8_all_st2lr0.5", m, "gain"), g("aw8_all_st2lr0.5", m, "lo"), g("aw8_all_st2lr0.5", m, "hi"))
        add("R3", "limits", "Time-blocked evaluation (hold out whole 100-frame blocks of a growth run; 25-frame buffer removed from training groups, pool and graph): the gain of lap3_k10_st4 over the tuned graph-free control survives (single 'blocked' split: AUC / accuracy)", "AUDIT (red-team development, 20 seeds): not a confirmation", "RT dev 8800-8819 (20 seeds)", fm("auc"), fm("acc"), ("not evaluated (blocked split has no anchor strata)", "n/a"),
            "vs lap3_k10: AUC " + L.fmt_gain(g("lap3_k10", "auc", "gain")) + ", acc " + L.fmt_gain(g("lap3_k10", "acc", "gain")) + "; vs shuffled graph: AUC " + L.fmt_gain(g("lap3_k10_st4_shuf", "auc", "gain")), evidence(["RT"], "(rt/a3)"), kind="audit")
    return ROWS


def appendix():
    return [
        ("E1", "Fine-tuned end-to-end model (Task 3c, 8 acquisition strategies, 5 seeds): strategies cannot be separated; training noise exceeds the differences", "DOCUMENTED ONLY", "code/ACADEMIC_REPORT_DRAFT.md §5.10"),
        ("E2", "Ideal-image type accuracy (28 outer-test images) is not an endpoint on which acquisition rules or encoders can be compared: reference anchors alone reach 0.848, simple controls reach 0.81-0.91", "DOCUMENTED ONLY (consistent with G6 regenerated above)", "code/ACADEMIC_REPORT_DRAFT.md §5.11, §5.19"),
        ("E3", "Held-out preference endpoint, old head schedule: no acquisition rule robustly beats random; significant rules differ by split and query unit", "DOCUMENTED ONLY", "code/ACADEMIC_REPORT_DRAFT.md §5.12-5.14"),
        ("E4", "Under a head schedule suited to 10-60 judgments, pool-wide variance-reduction rules (I-optimal, Fisher D-optimal, BALD x P(decisive)) beat random in pre-registered runs on fresh splits (about +0.01 to +0.02 accuracy, +0.006 to +0.02 AUC); the effect decays with the number of labels in hand", "DOCUMENTED ONLY (pre-registered runs of the earlier line; superseded by S1/S2: absorbed by the tuned learner)", "code/ACADEMIC_REPORT_DRAFT.md §5.15"),
        ("E5", "Type-aware graph coverage rule (typed_decisive_coverage) beats random under the re-tuned schedule (pooled 105 fresh seeds; log-loss +0.014 to +0.037, Holm < 0.01 per block); not distinguishable from the non-graph variance-reduction design", "DOCUMENTED ONLY (earlier learner; selection rules give about 0 with the tuned learners, see S3)", "code/ACADEMIC_REPORT_DRAFT.md §5.17-5.20"),
        ("E6", "The comparison graph (images = nodes, labelled pair groups = edges) is nearly a perfect matching (168 edges, 284 nodes, 117 components): no ranking structure for a GNN to recover", "DOCUMENTED ONLY", "code/ACADEMIC_REPORT_DRAFT.md §5.16; results/pair_graph_summary.csv"),
        ("E7", "About 60 graph constructions tried in development (mutual kNN, eps-graph, shared neighbours, Gaussian weights, PCA whitening, other layers, ImageNet features, label-aware topologies, purification): none better than the plain cosine kNN graph (k = 10) by more than +0.003 AUC; several 'purer' graphs are worse", "DOCUMENTED ONLY (development; ~60 variants in 13 rounds)", "graph_exploration/RESULTS_graph_mechanism.md §3"),
        ("E8", "Independent re-implementation of aw8_all / lap3_k10 / lap3_k10_st4 reproduces the main implementation's scores to <= 1e-15 on the audited cell; identity audit: no held-out image is a node/anchor/labelled image by path, one content twin in 3 of 10 audited cells", "DOCUMENTED ONLY (red-team audits 4 and 6)", "graph_exploration/PREREGISTRATION_RT1.md; new_directions/results/rt/a6.md"),
        ("E9", "Aggregation sensitivity: the lap3_k10_st4 gains over aw8_all_st2lr0.5 and lap3_k10 stay positive under budget-subset and median aggregations and sign-flip tests", "DOCUMENTED ONLY (red-team audit 7; table in new_directions/results/rt/a7.md)", "new_directions/results/rt/a7.md"),
    ]


def main():
    rows = build(); df = pd.DataFrame(rows).fillna(""); df.to_csv(L.TABLES / "claims.csv", index=False)
    esc = lambda s: str(s).replace("|", "/").replace("\n", " ")
    head = "| ID | Claim | Status | Seeds (35 each unless noted) | ΔAUC split A / B [95% CI] | Δacc split A / B [95% CI] | Anchor-far check (ΔAUC, both images > 0.03 from every anchor; RT dev 20 seeds) | Further blocks (ΔAUC point estimates A / B) | Evidence (branch@commit) |\n|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {r.id} | {esc(r.claim)} | **{esc(r.status)}** | {esc(r.seeds)} | {esc(r.dAUC)} | {esc(r.dACC)} | {esc(r.anchor_far)}{(' — ' + r.anchor_far_status) if r.anchor_far_status not in ('n/a', '') else ''} | {esc(r.replications)} | {esc(r.evidence)} |\n" for r in df.itertuples())
    short_status = lambda s: ("CONFIRMED" + (" + replicated" if "replicated in" in s and "NOT replicated" not in s and "PARTLY" not in s else (" (partly / not robustly replicated)" if "replicated" in s or "PARTLY" in s else ""))) if s.startswith("PRE-REGISTERED, CONFIRMED") else ("NOT CONFIRMED" if s.startswith("PRE-REGISTERED, NOT") else ("PRE-REG. rule met, practical claim not supported" if "PRACTICAL" in s else ("PRE-REG. control" if "CONTROL" in s else ("PRE-REG. rule: mixed (see full table)" if s.startswith("PRE-REGISTERED") else ("DESCRIPTIVE" if s.startswith("DESCRIPTIVE") else ("DEVELOPMENT only" if s.startswith("DEVELOPMENT") else ("AUDIT (red-team dev)" if s.startswith("AUDIT") else s)))))))
    pt = lambda x: re.sub(r" \[[^\]]*\]", "", str(x)); shorten = lambda t: (t.split(" (")[0].split(": ")[0] if len(t) > 100 else t)[:130]
    comp = "| ID | Claim | Status | ΔAUC A / B | Δacc A / B | Anchor-far ΔAUC A / B |\n|---|---|---|---|---|---|\n" + "".join(f"| {r.id} | {esc(SHORT.get(r.id, shorten(r.claim)))} | {short_status(r.status)} | {esc(pt(r.dAUC))} | {esc(pt(r.dACC))} | {esc(pt(r.anchor_far))} |\n" for r in df.itertuples())
    L.write(L.TABLES / "claims_onepage.md", "*Point estimates only (intervals, seeds and evidence in the full table below / `claims_table.md`). Splits A / B.*\n\n" + comp)
    note = ("*Gains are paired over seeds on identical label sets (mean over budgets 10/20/40/60 and 3 label draws); only AUC and accuracy are shown (log-loss is in `confirmations.csv`). Intervals: 95% bootstrap over seeds (the 168 labelled pair groups are the same in every seed: intervals and p-values are optimistic). "
            "ND = new-directions-overnight, GL = graph-learner, SH = selection-htr, GM = graph-mechanism, GE = graph-endpoint, RT = red-team.*\n\n")
    L.write(L.TABLES / "claims_table.md", note + head + body)
    ap = "| ID | Documented claim (numbers NOT regenerated in this package) | Status | Source |\n|---|---|---|---|\n" + "".join(f"| {a} | {esc(b)} | {c} | {d} |\n" for a, b, c, d in appendix()); L.write(L.TABLES / "claims_appendix.md", ap)
    N.save(); print(len(df), "claims +", len(appendix()), "documented-only rows")


if __name__ == "__main__":
    main()

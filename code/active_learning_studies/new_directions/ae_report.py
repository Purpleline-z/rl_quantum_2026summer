"""Generated tables and verdicts for the anchor-exchange study.  python3 ae_report.py --out TABLES.md [--template T --doc D]"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ae_analyze as aa

R = aa.R
CAND = ["htrq", "kcf", "kcg", "typi"]; CONTROL = {"htrq": "htrq_shuf", "kcg": "kcg_shuf"}; ALL_RULES = ["htrq", "htrq_shuf", "kcf", "kcg", "kcg_shuf", "typi", "amb", "amb_shuf"]
TYPE_METRICS = [("htr_acc", "HTR accuracy"), ("htr_auc", "HTR AUC"), ("htr_ll", "HTR log-loss"), ("t1x1_acc", "(1x1) accuracy"), ("tc6x2_acc", "c(6x2) accuracy"), ("t13_acc", "RT13 accuracy")]


def type_composition(frame: pd.DataFrame, rules) -> pd.DataFrame:
    f = frame[frame.budget == frame.budget.min()]; rows = []
    for r in rules:
        g = f[f.arm.str.startswith("rand")] if r == "rand" else f[f.arm == r]
        if g.empty: continue
        v = np.array([x for x in g.anchor_types if len(x) == 4], float); rows.append({"rule": r, "n_anchors": float(v.sum(1).mean()), "(1x1)": float(v[:, 0].mean()), "c(6x2)": float(v[:, 1].mean()), "RT13": float(v[:, 2].mean()), "HTR": float(v[:, 3].mean())})
    return pd.DataFrame(rows)


def active_block(name: str, title: str, rules, holm_rules):
    path = R / name
    if not path.exists() or not list(path.glob("seed*_*.json")): return f"### {title}\n\n(not run yet)\n\n", None
    f = aa.load(path); t = aa.active_table(f, holm_rules); text = f"### {title}\n\nseeds: {f.seed.nunique()}; anchors per arm: 36 of 116; gains over random anchors (mean of 3 random subsets), positive = better.\n\n"
    main = t[t.metric.isin(["auc", "acc"])]; text += "**Primary metrics (Holm over candidates x splits x {AUC, accuracy} = family of this block's listed rules)**\n\n" + aa.md(main[["rule", "split", "metric", "n_seeds", "gain", "share_better", "p", "holm"]])
    pairs = [(c, k) for c, k in CONTROL.items() if c in rules and k in set(f.arm)]; ctl = aa.control_table(f, pairs) if pairs else pd.DataFrame()
    text += "\n**Candidate minus its shuffled-graph control (uncorrected)**\n\n" + aa.md(ctl) + "\n"
    sec = t[t.metric.isin(["ll", "htr_acc", "htr_ll", "ext_acc"])]; text += "**Secondary: other endpoints (descriptive; no Holm)**\n\n" + aa.md(sec[["rule", "split", "metric", "gain", "share_better", "p"]]) + "\n"
    ext = t[t.metric == "ext_acc"].copy()
    if len(ext): hp = aa.holm({i: p for i, p in zip(ext.index, ext.p)}); ext["holm"] = [hp[i] for i in ext.index]; text += "**Secondary: external-test type accuracy (28 outer-test ideal images), Holm over rules x splits**\n\n" + aa.md(ext[["rule", "split", "n_seeds", "gain", "share_better", "p", "holm"]]) + "\n"
    text += "**Anchor type composition (mean number of anchors of each class)**\n\n" + aa.md(type_composition(f, ["rand"] + list(rules))) + "\n"
    return text, {"table": t, "frame": f, "ctl": ctl}


def verdict_a(res) -> str:
    if not res: return "not run yet"
    t, ctl = res["table"], res["ctl"]; parts = []
    for c in CAND:
        g = t[(t.rule == c) & t.metric.isin(["auc", "acc"])]
        if g.empty: continue
        pos = int((g.gain > 0).sum()); sig = int(((g.gain > 0) & (g.holm < .05)).sum())
        if c in CONTROL and len(ctl):
            cc = ctl[ctl.comparison == f"{c} - {CONTROL[c]}"]; beats = bool(((cc["diff"] > 0) & (cc.p_uncorrected < .05)).any())
        else: beats = None
        ok = pos == 4 and sig >= 3 and (beats is not False)
        parts.append(f"{c}: {pos}/4 gains positive, {sig} Holm<0.05, beats shuffled control: {'n/a (graph-free)' if beats is None else beats} -> {'SUCCESS' if ok else 'not confirmed'}")
    return "; ".join(parts)


def dose_block(name: str, title: str, reps: int = 1000):
    path = R / name
    if not path.exists() or not list(path.glob("seed*_*.json")): return f"### {title}\n\n(not run yet)\n\n", None
    f = aa.load(path); text = f"### {title}\n\nseeds: {f.seed.nunique()}; anchors n in {{0, 12, 29, 58, 116}} (random subsets, mean over 3 subsets) x pairwise budgets (extra judgments beyond the 10 initial groups). Values = mean over seeds and splits; 'gain' = minus the 0-anchor row of the same budget.\n\n"
    for m, label in (("auc", "AUC"), ("acc", "accuracy"), ("ll", "−log-loss (higher = better)"), ("ext_acc", "external-test type accuracy")) + tuple(TYPE_METRICS): text += f"**{label}**\n\n" + aa.md(aa.dose_table(f, m)) + "\n"
    arm = f[f.arm == "all144"]
    if len(arm):
        base = f[(f.arm != "all144") & (f.n_anchors == 116)].groupby(["seed", "split", "budget"], as_index=False)[["auc", "acc"]].mean(); a144 = arm.groupby(["seed", "split", "budget"], as_index=False)[["auc", "acc"]].mean(); m = a144.merge(base, on=["seed", "split", "budget"], suffixes=("_144", "_116"))
        if len(m): text += "**Cost of dropping the 28 outer-test anchors (144 anchors − 116 anchors, same learner)**\n\n" + aa.md(pd.DataFrame([{"metric": k, "diff_144_minus_116": float((m[f"{k}_144"] - m[f"{k}_116"]).mean()), "p_uncorrected": aa.wil((m[f"{k}_144"] - m[f"{k}_116"]).groupby([m.seed, m.split]).mean())} for k in ("auc", "acc")])) + "\n"
    trans = [(0, 12), (12, 29), (29, 58), (58, 116)]; b0s = [10, 20, 40]
    text += "**Exchange rate (overall AUC): pairwise judgments equivalent to the extra anchors, with seed-bootstrap 95% interval**\n\n" + aa.md(aa.exchange_table(f, "auc", trans, b0s, reps=reps)) + "\n"
    text += "**Exchange rate (overall accuracy)**\n\n" + aa.md(aa.exchange_table(f, "acc", trans, b0s, reps=reps)) + "\n"
    for m, label in TYPE_METRICS: text += f"**Exchange rate, {label} (per type)**\n\n" + aa.md(aa.exchange_table(f, m, trans, b0s, reps=reps)) + "\n"
    for split in ("A", "B"): text += f"**Exchange rate (overall AUC), split {split} only**\n\n" + aa.md(aa.exchange_table(f, "auc", [(0, 29), (29, 116)], [20], split=split, reps=reps)) + "\n"
    return text, {"frame": f}


def lam_block(name: str, title: str):
    path = R / name
    if not path.exists() or not list(path.glob("seed*_*.json")): return f"### {title}\n\n(not run yet)\n\n"
    f = aa.load(path); g = f.groupby(["split", "seed", "n_anchors", "lam"], as_index=False)[["auc", "acc", "ll"]].mean(); rows = []
    for (n, lam), gg in g.groupby(["n_anchors", "lam"]):
        ref = g[(g.n_anchors == n) & (g.lam == 3.0)].set_index(["split", "seed"]); this = gg.set_index(["split", "seed"]); d = this[["auc", "acc"]] - ref[["auc", "acc"]]
        rows.append({"anchors": n, "lambda": lam, "auc": float(this.auc.mean()), "acc": float(this.acc.mean()), "auc_minus_lambda3": float(d.auc.mean()), "acc_minus_lambda3": float(d.acc.mean())})
    base = g[(g.n_anchors == 116) & (g.lam == 3.0)]; return f"### {title}\n\nseeds: {f.seed.nunique()}; steps x4 in all rows; lambda = 0 is the Laplacian-free learner with the same schedule.\n\n" + aa.md(pd.DataFrame(rows)) + "\n"


def sections(reps: int = 1000) -> dict:
    sec = {}
    sec["dev_active"], r = active_block("dev_active", "Development: active anchor rules at 36 anchors (seeds 8700–8713; NOT evidence)", ALL_RULES, [x for x in ALL_RULES])
    sec["conf_active"], rc = active_block("conf_active", "Confirmation 1: active anchor rules at 36 anchors (seeds 8750–8784)", CAND + list(CONTROL.values()), CAND)
    sec["dev_dose"], _ = dose_block("dev_dose", "Development: anchor dose-response and exchange rate (seeds 8700–8709; NOT evidence)", reps)
    sec["conf_dose"], _ = dose_block("conf_dose", "Confirmation 1: anchor dose-response and exchange rate (seeds 8750–8784)", reps)
    sec["dev_lam"] = lam_block("dev_lam", "Development: interaction with lambda (seeds 8700–8709; NOT evidence)")
    sec["exploratory"] = ""
    if (R / "dev_active2").exists() and list((R / "dev_active2").glob("seed*_*.json")):
        t, _ = active_block("dev_active2", "Exploratory: class balance of the anchors (dev seeds 8700–8719; strat = class-balanced random, a DIAGNOSTIC that uses the true classes; kcf_bal = sequential k-center with class balance)", ["strat", "kcf_bal", "kcf"], ["strat", "kcf_bal", "kcf"]); sec["exploratory"] += t
    ans = Path(__file__).resolve().parent.parent / "pair_disjoint_not_image_disjoint" / "graph_exploration" / "answers_anchor_exchange.md"; sec["answers"] = ans.read_text() if ans.exists() else "(answers not written yet)"
    sec["verdict"] = f"- **Confirmation 1, rule (a)**: {verdict_a(rc)}\n"
    return sec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--template", default=None); ap.add_argument("--doc", default=None); ap.add_argument("--reps", type=int, default=1000); a = ap.parse_args(); sec = sections(a.reps)
    if a.out:
        text = "# Tables for the anchor-exchange study (generated by ae_report.py; do not edit by hand)\n\n## Pre-registered reading\n\n" + sec["verdict"] + "\n## Confirmation\n\n" + sec["conf_active"] + sec["conf_dose"] + "\n## Development (not evidence)\n\n" + sec["dev_active"] + sec["dev_dose"] + sec["dev_lam"]
        Path(a.out).write_text(text); print(f"wrote {a.out} ({len(text)} chars)")
    if a.template and a.doc:
        t = Path(a.template).read_text()
        for k, v in sec.items(): t = t.replace(f"<!--TABLE:{k}-->", v).replace(f"<!--SECTION:{k}-->", v)
        Path(a.doc).write_text(t); print(f"wrote {a.doc}")


if __name__ == "__main__":
    main()

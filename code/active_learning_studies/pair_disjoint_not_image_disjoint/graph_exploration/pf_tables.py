"""Tables of the physics-features study, generated from the per-judgment result files of ``new_directions/pf_run.py`` (nothing typed by hand).

  python3 pf_tables.py dev DIR.. [--ref=lap3_k10_st4] [--only=a,b]       gain of every learner over the reference: standard endpoint (AUC, accuracy, log-loss, HTR accuracy) and the anchor-far stratum (AUC, accuracy)
  python3 pf_tables.py confirm SPEC.json DIR..                           pre-registered tests (families and verdict rule in the spec / PREREGISTRATION_PF1.md)
  python3 pf_tables.py diff DIR.. --pairs=a:b,c:d                        exploratory paired differences (uncorrected)
Definitions: a judgment is "anchor-far" if the minimum over its two images of the cosine distance (centred SimCLR features) to the nearest typed ideal image (all anchors of aw8_all) is > 0.01 (as in the red-team audit).
Per (seed, split, budget, draw) metrics are computed on all decisive held-out judgments (confidence-weighted accuracy and log-loss, AUC) and on the anchor-far subset (AUC only if both outcomes occur); unit of analysis = per-seed mean of the paired difference
over budgets and draws; two-sided Wilcoxon signed-rank over seeds, bootstrap 95 % interval, Holm within the named family."""
import glob, json, sys, hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

FAR = 0.01
CACHE = Path("/tmp/pf_tables_cache"); CACHE.mkdir(exist_ok=True)


def _cell_metrics(g: pd.DataFrame) -> dict:
    s, d, w = g.sign.to_numpy(), g.d.to_numpy(), g.weight.to_numpy(); far = (g.anc_min > FAR).to_numpy(); out = {}
    def block(m, pre):
        if not m.any(): return {}
        r = {f"{pre}acc": float((w[m] * ((d[m] * s[m]) > 0)).sum() / w[m].sum()), f"{pre}ll": float((w[m] * np.logaddexp(0.0, -s[m] * d[m])).sum() / w[m].sum()), f"{pre}n": int(m.sum())}
        r[f"{pre}auc"] = float(roc_auc_score(s[m] > 0, d[m])) if len(set(s[m])) == 2 else np.nan; return r
    out.update(block(np.ones(len(d), bool), "")); out.update(block(far, "far_")); out.update(block(~far, "near_")); h = (g.k == 4).to_numpy()
    if h.any(): out["htr_acc"] = float(((d[h] * s[h]) > 0).mean()); out["htr_ll"] = float(np.logaddexp(0.0, -s[h] * d[h]).mean())
    for k, name in ((0, "t1x1"), (2, "tc6x2"), (3, "t13")):
        m = (g.k == k).to_numpy()
        if m.any(): out[f"{name}_acc"] = float(((d[m] * s[m]) > 0).mean())
    return out


def load(dirs) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "seed*.csv.gz"))):
            key = CACHE / (hashlib.md5((f + str(Path(f).stat().st_mtime) + str(Path(f).stat().st_size)).encode()).hexdigest() + ".pkl")
            if key.exists(): frames.append(pickle.loads(key.read_bytes())); continue
            df = pd.read_csv(f); rows = [{"seed": a[0], "split": a[1], "budget": a[2], "draw": a[3], "learner": a[4], **_cell_metrics(g)} for a, g in df.groupby(["seed", "split", "budget", "draw", "learner"])]
            r = pd.DataFrame(rows); key.write_bytes(pickle.dumps(r)); frames.append(r)
    f = pd.concat(frames, ignore_index=True)
    # a file may be rewritten with more learners; drop exact duplicates, keep complete (seed, split) cells only: every learner present in the directory's fullest file has all (budget, draw) cells
    return f.drop_duplicates(["seed", "split", "budget", "draw", "learner"])


SIGN = {"auc": 1, "acc": 1, "ll": -1, "far_auc": 1, "far_acc": 1, "far_ll": -1, "near_auc": 1, "near_acc": 1, "htr_acc": 1, "htr_ll": -1, "t1x1_acc": 1, "tc6x2_acc": 1, "t13_acc": 1}


def per_seed_gain(frame, splits, metric, a, b):
    """Series indexed by seed: mean over the selected splits, budgets, draws of SIGN * (a - b), paired on (seed, split, budget, draw)."""
    f = frame[frame.split.isin(splits)]; ka = f[f.learner == a].set_index(["seed", "split", "budget", "draw"])[metric]; kb = f[f.learner == b].set_index(["seed", "split", "budget", "draw"])[metric]
    d = (SIGN[metric] * (ka - kb)).dropna(); return d.groupby(level=0).mean()


def summarize(per: pd.Series, n_boot=4000) -> dict:
    v = per.dropna().to_numpy(); rng = np.random.default_rng(0)
    if len(v) < 3: return {"gain": float(v.mean()) if len(v) else np.nan, "lo": np.nan, "hi": np.nan, "share": np.nan, "p": np.nan, "n": len(v)}
    b = rng.choice(v, (n_boot, len(v))).mean(1)
    try: p = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    except ValueError: p = 1.0
    return {"gain": float(v.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "share": float((v > 0).mean()), "p": p, "n": len(v)}


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1]); m = len(items); out, run = {}, 0.0
    for i, (k, p) in enumerate(items): run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def to_md(df, fmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def dev_table(frame, ref, only=None, metrics=("auc", "acc", "ll", "htr_acc", "far_auc", "far_acc")) -> pd.DataFrame:
    rows = []
    for split in sorted(frame.split.unique()):
        f = frame[frame.split == split]; learners = sorted(set(f.learner) - {ref}); learners = [l for l in learners if (not only or l in only)]
        for l in learners:
            r = {"split": split, "learner": l}
            for m in metrics:
                s = summarize(per_seed_gain(frame, [split], m, l, ref)); r[m] = s["gain"]
            r["n"] = s["n"]; rows.append(r)
    return pd.DataFrame(rows)


def confirm_tables(spec, frame):
    """Main family: candidate - reference, {A, B} x {AUC, accuracy} (Holm over candidates x 4).  Anchor-far AUC (pooled over both splits, per-seed mean) with its 95 % interval is part of the success rule.  Controls (uncorrected):
    candidate - shuffled-graph control and candidate - random-projection control (where present).  Secondary / exploratory: log-loss, HTR accuracy, far accuracy, candidate - graph-free control, reference - graph-free control."""
    ref = spec["reference"]; splits = spec["splits"]; rows = []
    def add(family, cand, a, b, split_list, metric, label):
        s = summarize(per_seed_gain(frame, split_list, metric, a, b)); rows.append({"family": family, "cand": cand, "cells": f"{a} - {b}", "split": label, "metric": metric, **s})
    for cand, ctl in spec["candidates"].items():
        for sp in splits:
            for m in ("auc", "acc"): add("main", cand, cand, ref, [sp], m, sp)
        add("far", cand, cand, ref, splits, "far_auc", "A+B")
        for sp in splits: add("far_split", cand, cand, ref, [sp], "far_auc", sp)
        for sp in splits:
            for m in ("ll", "htr_acc", "far_acc", "near_auc"): add("secondary", cand, cand, ref, [sp], m, sp)
        for k, name in ctl.items():
            if name in set(frame.learner):
                for sp in splits:
                    for m in ("auc", "acc"): add(f"control:{k}", cand, cand, name, [sp], m, sp)
                add(f"control_far:{k}", cand, cand, name, splits, "far_auc", "A+B")
        for sp in splits:
            for m in ("auc", "acc"): add("vs_graphfree", cand, cand, spec["graphfree"], [sp], m, sp)
    for sp in splits:
        for m in ("auc", "acc", "far_auc"): add("baseline_vs_graphfree", ref, ref, spec["graphfree"], [sp], m, sp)
    out = pd.DataFrame(rows)
    for fam in ("main",):
        sel = out[out.family == fam]; h = holm({i: p for i, p in sel.p.items() if not np.isnan(p)}); out.loc[h.keys(), "holm"] = list(h.values())
    verdict = []
    for cand in spec["candidates"]:
        main = out[(out.family == "main") & (out.cand == cand)]; far = out[(out.family == "far") & (out.cand == cand)].iloc[0]; allpos = bool((main.gain > 0).all() and len(main) == 4); nsig = int(((main.holm < 0.05) & (main.gain > 0)).sum())
        far_ok = bool(far.gain > 0 and far.lo > 0); ctl_ok = {}
        for k in spec["candidates"][cand]:
            cc = out[(out.family == f"control:{k}") & (out.cand == cand)]; ctl_ok[k] = bool(((cc.gain > 0) & (cc.p < 0.05)).any()) if len(cc) else None
        verdict.append({"candidate": cand, "all_four_positive": allpos, "n_holm_sig": nsig, "far_auc_gain": float(far.gain), "far_auc_lo": float(far.lo), "far_ok": far_ok, **{f"beats_{k}": v for k, v in ctl_ok.items()},
                        "CONFIRMED": bool(allpos and nsig >= 3 and far_ok and all(v is True for v in ctl_ok.values()))})
    return out, pd.DataFrame(verdict)


if __name__ == "__main__":
    mode = sys.argv[1]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); args = [a for a in sys.argv[2:] if not a.startswith("--")]
    if mode == "dev":
        frame = load(args); only = opts["only"].split(",") if "only" in opts else None; ref = opts.get("ref", "lap3_k10_st4"); print(f"#### gain over `{ref}` (AUC/accuracy/HTR accuracy: learner - ref; log-loss: ref - learner; mean over budgets, draws, then seeds); far_* = anchor-far stratum\n"); print(to_md(dev_table(frame, ref, only)))
    elif mode == "confirm":
        spec = json.loads(Path(args[0]).read_text()); frame = load(args[1:]); out, ver = confirm_tables(spec, frame); cols = ["family", "split", "metric", "cells", "n", "gain", "lo", "hi", "share", "p", "holm"]
        print(to_md(out[cols].astype({c: float for c in ("gain", "lo", "hi", "share", "p", "holm")}))); print(); print(to_md(ver, "{:+.4f}"))
    elif mode == "diff":
        frame = load(args); rows = []
        for pr in opts["pairs"].split(","):
            a, b = pr.split(":")
            for split in sorted(frame.split.unique()):
                for m in ("auc", "acc", "ll", "htr_acc", "far_auc", "far_acc", "near_auc", "t1x1_acc", "tc6x2_acc", "t13_acc"):
                    s = summarize(per_seed_gain(frame, [split], m, a, b)); rows.append({"pair": f"{a} - {b}", "split": split, "metric": m, **s})
        print(to_md(pd.DataFrame(rows)))

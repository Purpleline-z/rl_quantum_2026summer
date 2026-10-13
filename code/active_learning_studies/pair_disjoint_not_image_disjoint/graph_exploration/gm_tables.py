"""Tables of the graph-mechanism study (generated from result JSONs; nothing typed by hand).

  python3 gm_tables.py dev DIR [DIR..] [--refs=aw8_all,lap3_k10] [--only=a,b]   gain of every learner over each reference (AUC, accuracy, log-loss, HTR accuracy), per split
  python3 gm_tables.py dose DIR [DIR..] [--seeds=LO-HI]                          graph-quality dose-response: per-seed Spearman(gain over aw8_all, graph quality) across the noise doses, per lambda
  python3 gm_tables.py confirm SPEC.json DIR [DIR..]                            pre-registered tests of PREREGISTRATION_GM1 (see spec)
  python3 gm_tables.py types DIR.. --learners=a,b [--ref=lap3_k10]  /  budget ...  per-type / per-budget gains
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
NEW = HERE.parents[1] / "new_directions"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(NEW)); sys.path.insert(0, str(NEW.parents[1] / "active_learning_program"))
import gl_tables as gt  # noqa: E402
import nd_confirm as c  # noqa: E402

QUALITY = NEW / "results" / "gm" / "quality"
DOSE = {3: [("gm_lap3_n0", 0.0), ("gm_lap3_n0.5", .5), ("gm_lap3_n1", 1.0), ("gm_lap3_n2", 2.0), ("gm_lap3_n3", 3.0)],
        10: [("gm_lap10_n0", 0.0), ("gm_lap10_n0.5", .5), ("gm_lap10_n1", 1.0), ("gm_lap10_n2", 2.0), ("gm_lap10_n3", 3.0)]}


def arg(prefix, default=None):
    for a in sys.argv[2:]:
        if a.startswith(prefix + "="): return a.split("=", 1)[1]
    return default


def dirs():
    return [a for a in sys.argv[2:] if not a.startswith("--")]


def quality_of(seed, split, name):
    f = QUALITY / f"seed{seed}_{split}.json"
    if not f.exists(): return np.nan
    d = json.loads(f.read_text()); return d.get(name, {}).get("quality", np.nan)


def per_seed_gain(frame, split, metric, learner, ref):
    return gt.gain(frame, split, metric, learner, ref)


def dose_table(frame, seeds=None, ref="aw8_all"):
    """Per (split, lambda, metric): mean quality and mean gain per dose; per-seed Spearman across doses, its mean and a Wilcoxon p over seeds (H1: rho != 0), plus the mean within-seed OLS slope."""
    rows_dose, rows_rho = [], []
    for split in sorted(frame.split.unique()):
        for lam, doses in DOSE.items():
            if not all(n in set(frame.learner) for n, _ in doses): continue
            for metric in ("auc", "acc", "ll"):
                gains = {n: per_seed_gain(frame, split, metric, n, ref) for n, _ in doses}; seeds_ = sorted(set.intersection(*[set(g.index) for g in gains.values()]))
                if seeds is not None: seeds_ = [s for s in seeds_ if s in seeds]
                rhos, slopes = [], []
                for s in seeds_:
                    g = np.array([gains[n][s] for n, _ in doses]); q = np.array([quality_of(s, split, n) for n, _ in doses])
                    if np.isnan(q).any() or np.ptp(q) == 0 or np.ptp(g) == 0: continue
                    rhos.append(spearmanr(q, g)[0]); slopes.append(np.polyfit(q, g, 1)[0])
                for n, sg in doses:
                    rows_dose.append({"split": split, "lambda": lam, "metric": metric, "sigma": sg, "learner": n, "quality": np.nanmean([quality_of(s, split, n) for s in seeds_]), "gain": float(np.mean([gains[n][s] for s in seeds_]))})
                if rhos:
                    try: p = float(wilcoxon(rhos).pvalue) if np.any(np.array(rhos) != 0) else 1.0
                    except ValueError: p = 1.0
                    rows_rho.append({"split": split, "lambda": lam, "metric": metric, "n_seeds": len(rhos), "mean_spearman": float(np.mean(rhos)), "share_rho_positive": float(np.mean(np.array(rhos) > 0)), "p_wilcoxon": p, "mean_slope_per_unit_quality": float(np.mean(slopes))})
    return pd.DataFrame(rows_dose), pd.DataFrame(rows_rho)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        frame = gt.load(dirs()); only = arg("--only"); refs = (arg("--refs") or "aw8_all,lap3_k10").split(",")
        for ref in refs:
            t = gt.dev_table(frame, ref, metrics=("auc", "acc", "ll", "htr_acc")); t = t[[x for x in t.columns if not x.endswith(("_lo", "_hi", "_vs_ctl", "_vs_ctl_p"))]]
            if only: t = t[t.learner.isin(only.split(","))]
            print(f"\n#### gain over `{ref}` (development seeds; AUC/accuracy: learner - ref; log-loss: ref - learner)\n"); print(gt.to_md(t))
    elif mode == "dose":
        frame = gt.load(dirs()); sd = arg("--seeds"); seeds = None
        if sd: lo, hi = sd.split("-"); seeds = set(range(int(lo), int(hi) + 1))
        d, r = dose_table(frame, seeds); print("\n#### graph quality and gain over aw8_all by noise dose\n"); print(gt.to_md(d)); print("\n#### per-seed Spearman(gain, graph quality) across the five doses\n"); print(gt.to_md(r))
    elif mode == "dosetest":   # pre-registered dose-response tests: SPEC json {"lambdas": [10], "metrics": ["auc","acc"], "splits": ["A","B"], "seeds": "8100-8134"}; Holm over the family
        spec = json.loads(Path(sys.argv[2]).read_text()); frame = gt.load([a for a in sys.argv[3:] if not a.startswith("--")]); lo, hi = spec["seeds"].split("-"); seeds = set(range(int(lo), int(hi) + 1))
        d, r = dose_table(frame, seeds); r = r[r["lambda"].isin(spec["lambdas"]) & r.metric.isin(spec["metrics"]) & r.split.isin(spec["splits"])].copy()
        r["holm"] = c.holm(r.p_wilcoxon.values); r["positive_and_sig"] = (r.mean_spearman > 0) & (r.holm < 0.05); print(gt.to_md(r)); print(); print(f"dose-response claim (>= {spec['min_sig']} of {len(r)} tests positive and Holm < .05): {'CONFIRMED' if int(r.positive_and_sig.sum()) >= spec['min_sig'] else 'NOT confirmed'}")
        print(); print(gt.to_md(d[d.metric.isin(spec["metrics"]) & d["lambda"].isin(spec["lambdas"])]))
    elif mode == "confirm":   # pre-registered tests of PREREGISTRATION_GM1 (spec: reference, candidates {name: {shuf, noise}}, nongraph, aw8)
        spec = json.loads(Path(sys.argv[2]).read_text()); frame = gt.load([a for a in sys.argv[3:] if not a.startswith("--")]); ref, aw8, ng = spec["reference"], spec["aw8"], spec["nongraph"]; T = []
        cell = lambda l: ["random", l]
        for cand, ctl in spec["candidates"].items():
            for split in spec["splits"]:
                for m in ("auc", "acc"):
                    T.append({"family": "main", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
                    T.append({"family": "vs_aw8", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(aw8)})
                    for k, name in ctl.items(): T.append({"family": f"control:{k}", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(name)})
                    T.append({"family": "control:nongraph", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ng)})
                for m in ("ll", "htr_acc", "htr_ll"): T.append({"family": f"secondary:{m}", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
        out = c.run({"tests": T}, frame.assign(selector="random")); out["cells"] = [f"{a[1]} - {b[1]}" for a, b in zip(out.a, out.b)]
        cols = ["family", "split", "metric", "cells", "n_seeds", "gain", "lo", "hi", "share_better", "p", "holm"]; print(gt.to_md(out[cols].assign(**{k: out[k].astype(float) for k in ("gain", "lo", "hi", "share_better", "p", "holm")})))
        rows = []
        for cand in spec["candidates"]:
            main = out[(out.family == "main") & (out.cand == cand)]; allpos = bool((main.gain > 0).all() and len(main) == 4); nsig = int((main.holm < 0.05).sum()); ctl_ok = {}
            for k in spec["candidates"][cand]:
                cc = out[(out.family == f"control:{k}") & (out.cand == cand)]; ctl_ok[k] = bool(((cc.gain > 0) & (cc.p < 0.05)).any())
            rows.append({"candidate": cand, "all_four_positive": allpos, "n_holm_sig": nsig, **{f"beats_{k}": v for k, v in ctl_ok.items()}, "CONFIRMED": bool(allpos and nsig >= 3 and all(ctl_ok.values()))})
        print(); print(gt.to_md(pd.DataFrame(rows), "{}"))
    elif mode == "diff":   # exploratory paired differences between named learners (uncorrected p): --pairs=a:b,c:d
        frame = gt.load(dirs()); rows = []
        for pr in arg("--pairs").split(","):
            a, b = pr.split(":")
            for split in sorted(frame.split.unique()):
                for m in ("auc", "acc", "ll", "htr_acc", "t1x1_acc", "tc6x2_acc", "t13_acc"):
                    try: s = gt.summarize(gt.gain(frame, split, m, a, b)); rows.append({"pair": f"{a} - {b}", "split": split, "metric": m, "gain": s["gain"], "lo": s["lo"], "hi": s["hi"], "share_better": s["share"], "p_uncorrected": s["p"], "n": s["n"]})
                    except Exception: pass
        print(gt.to_md(pd.DataFrame(rows)))
    elif mode == "curve":   # mean held-out AUC / accuracy by budget (levels, not gains): how many extra random labels would aw8_all need to reach the regularised learner?
        frame = gt.load(dirs()); names = arg("--learners").split(","); rows = []
        for split in sorted(frame.split.unique()):
            g = frame[frame.split == split]
            for l in names:
                for m in ("auc", "acc"):
                    v = g[g.learner == l].groupby(["seed", "budget"])[m].mean().groupby("budget").mean(); rows.append({"split": split, "learner": l, "metric": m, **{f"b{b}": float(x) for b, x in v.items()}})
        print(gt.to_md(pd.DataFrame(rows), "{:.4f}"))
    elif mode == "qgain":   # across-learner dose-response: per-seed Spearman(gain over aw8_all, graph quality) across the named learners (different graph constructions)
        frame = gt.load(dirs()); names = arg("--names").split(","); ref = arg("--ref", "aw8_all"); rows = []
        for split in sorted(frame.split.unique()):
            for metric in ("auc", "acc"):
                g = {n: per_seed_gain(frame, split, metric, n, ref) for n in names}; seeds_ = sorted(set.intersection(*[set(v.index) for v in g.values()])); rhos = []
                for s in seeds_:
                    q = np.array([quality_of(s, split, n) for n in names]); y = np.array([g[n][s] for n in names])
                    if not np.isnan(q).any(): rhos.append(spearmanr(q, y)[0])
                p = float(wilcoxon(rhos).pvalue) if len(rhos) > 5 else float("nan"); rows.append({"split": split, "metric": metric, "n_learners": len(names), "n_seeds": len(rhos), "mean_spearman": float(np.mean(rhos)), "p_wilcoxon": p})
        print(gt.to_md(pd.DataFrame(rows)))
        tab = []
        for n in names:
            r = {"learner": n}
            for split in sorted(frame.split.unique()): r[f"quality_{split}"] = float(np.nanmean([quality_of(s, split, n) for s in range(8000, 8020)])); r[f"auc_gain_{split}"] = float(per_seed_gain(frame, split, "auc", n, ref).mean())
            tab.append(r)
        print(); print(gt.to_md(pd.DataFrame(tab)))
    elif mode in ("budget", "types"):
        cands = arg("--learners").split(","); frame = gt.load(dirs()); ref = arg("--ref", "aw8_all")
        print(gt.to_md(gt.budget_table(frame, cands, ref) if mode == "budget" else gt.type_table(frame, cands, ref)))

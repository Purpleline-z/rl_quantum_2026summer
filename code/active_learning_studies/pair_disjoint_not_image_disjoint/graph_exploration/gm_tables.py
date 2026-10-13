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
    elif mode in ("budget", "types"):
        cands = arg("--learners").split(","); frame = gt.load(dirs()); ref = arg("--ref", "aw8_all")
        print(gt.to_md(gt.budget_table(frame, cands, ref) if mode == "budget" else gt.type_table(frame, cands, ref)))

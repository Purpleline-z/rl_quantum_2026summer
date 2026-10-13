"""Tables of the regulariser study (G2), generated from the result JSONs (nothing typed by hand).

  python3 rg_tables.py dev DIR [DIR ...] [--ref=lap3_k10_st4] [--ref2=aw8_all_st2lr0.5]    every learner vs the references on seeds x budgets (paired, identical label sets), by split:
                                                                                         AUC, accuracy, log-loss, anchor-far AUC / accuracy, anchor-near AUC / accuracy (uncorrected Wilcoxon p)
  python3 rg_tables.py confirm SPEC.json DIR [DIR ...]                                   the pre-registered tests (non-inferiority, superiority, controls, anchor-far secondary), Holm within family
  python3 rg_tables.py budgets DIR [...] [--ref=...]                                     gain vs the reference by budget (descriptive)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

SIGN = {"auc": 1, "acc": 1, "ll": -1, "auc_far": 1, "acc_far": 1, "ll_far": -1, "auc_near": 1, "acc_near": 1, "ll_near": -1}
METRICS = ("auc", "acc", "ll", "auc_far", "acc_far", "auc_near", "acc_near")


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def load(dirs):
    """All records of the directories; files that are still being written (fewer records than the fullest file of their directory) are skipped."""
    rows = []
    for d in dirs:
        files = {f: json.loads(f.read_text()) for f in sorted(Path(d).glob("seed*_*.json"))}
        if not files: continue
        full = max(len(v) for v in files.values()); rows += [{k: v for k, v in r.items() if k != "d"} for v in files.values() if len(v) == full for r in v]
    return pd.DataFrame(rows)


def per_seed(frame, split, metric, a, b):
    """Per seed: mean over budgets (and draws) of sign * (metric_a - metric_b), paired on (seed, budget, draw); NaN cells (a subset without both classes) are dropped pairwise."""
    g = frame[frame.split == split]; ga = g[g.learner == a].set_index(["seed", "budget", "draw"])[metric]; gb = g[g.learner == b].set_index(["seed", "budget", "draw"])[metric]
    return (SIGN[metric] * (ga - gb)).dropna().groupby(level="seed").mean()


def wil(per, shift=0.0, alternative="two-sided"):
    x = per.to_numpy() + shift
    try: return float(wilcoxon(x, alternative=alternative).pvalue) if np.any(x != 0) else 1.0
    except ValueError: return 1.0


def summarize(per):
    rng = np.random.default_rng(0); b = rng.choice(per.to_numpy(), (3000, len(per))).mean(1)
    return {"gain": per.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5), "share_better": float((per > 0).mean()), "n_seeds": len(per)}


def fmt_md(df, floatfmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def dev_table(frame, ref, metrics=METRICS):
    rows = []
    for split in sorted(frame.split.unique()):
        for l in sorted(set(frame[frame.split == split].learner) - {ref}):
            r = {"split": split, "learner": l}
            for m in metrics:
                try: p = per_seed(frame, split, m, l, ref); r["n"] = len(p); r[m] = p.mean(); r[m + "_p"] = wil(p)
                except Exception: pass
            rows.append(r)
    return pd.DataFrame(rows)


def short(df):
    keep = [c for c in df.columns if not c.endswith("_p")]; return df[keep]


def build_tests(spec):
    """Spec keys: reference (the graph learner lap3_k10_st4), tuned (graph-free tuned baseline), freecands (graph-free candidates), graphcands (further graph learners), controls {graph learner: its
    shuffled-graph control}, margins {auc, acc}, splits.  Families:
      NI:<cand>        non-inferiority of a graph-free candidate vs the reference: one-sided Wilcoxon on (cand - ref) + margin, splits x {auc, acc}   (success A; Holm over all NI tests of the main endpoint)
      SUP:<G>:<cand>   superiority of the graph learner G (the reference, or a graph candidate) over a graph-free candidate (G - cand), two-sided Wilcoxon, splits x {auc, acc}
                       (success B; Holm within the 'SUP:<G>' set = all graph-free candidates x splits x metrics)
      CTRL:<G>         graph learner vs its own shuffled-graph control (G - control), two-sided, uncorrected
      TUNED            reference vs the tuned graph-free baseline aw8_all_st2lr0.5, two-sided (re-check of the earlier confirmation; uncorrected)
      FAR_NI / FAR_SUP the same NI / SUP(reference) tests on the anchor-far endpoint (auc_far, acc_far); secondary, not part of any success rule"""
    T = []; ref = spec["reference"]; graphs = [ref] + spec.get("graphcands", [])
    for c in spec["freecands"]:
        for split in spec["splits"]:
            for m in ("auc", "acc"):
                T.append({"family": f"NI:{c}", "kind": "NI", "cand": c, "split": split, "metric": m, "a": c, "b": ref, "margin": spec["margins"][m]})
                for g in graphs: T.append({"family": f"SUP:{g}", "kind": "SUP", "cand": c, "graph": g, "split": split, "metric": m, "a": g, "b": c})
            for m in ("auc_far", "acc_far"):
                T.append({"family": "FAR_NI", "kind": "FAR_NI", "cand": c, "split": split, "metric": m, "a": c, "b": ref, "margin": spec["margins"][m.split("_")[0]]})
                T.append({"family": "FAR_SUP", "kind": "FAR_SUP", "cand": c, "graph": ref, "split": split, "metric": m, "a": ref, "b": c})
    for g, ctl in spec.get("controls", {}).items():
        for split in spec["splits"]:
            for m in ("auc", "acc"): T.append({"family": f"CTRL:{g}", "kind": "CTRL", "cand": g, "split": split, "metric": m, "a": g, "b": ctl})
    for split in spec["splits"]:
        for m in ("auc", "acc"): T.append({"family": "TUNED", "kind": "TUNED", "cand": ref, "split": split, "metric": m, "a": ref, "b": spec["tuned"]})
    return T


def run_tests(frame, spec):
    rows = []
    for t in build_tests(spec):
        per = per_seed(frame, t["split"], t["metric"], t["a"], t["b"])
        p = wil(per, shift=t["margin"], alternative="greater") if t["kind"] in ("NI", "FAR_NI") else wil(per)
        rows.append({**t, **summarize(per), "p": p})
    out = pd.DataFrame(rows); out["holm"] = np.nan
    m = out.kind == "NI"; out.loc[m, "holm"] = holm(out.loc[m, "p"])                                    # Holm over all NI tests (candidates x splits x {auc, acc})
    for g in out[out.kind == "SUP"].graph.unique(): m = (out.kind == "SUP") & (out.graph == g); out.loc[m, "holm"] = holm(out.loc[m, "p"])   # Holm within each graph learner's SUP set
    for k in ("FAR_NI", "FAR_SUP"): m = out.kind == k; out.loc[m, "holm"] = holm(out.loc[m, "p"])
    return out


def verdict(out, spec):
    """Success A (graph not needed): some graph-free candidate has >= 3 of its 4 non-inferiority tests Holm < 0.05 (one-sided; margins 0.003 AUC / 0.01 accuracy).
    Success B (graph adds): some graph learner G (the reference or a graph candidate) beats EVERY graph-free candidate: for each candidate all four differences G - cand positive and >= 3 of 4 Holm < 0.05.
    Both can hold (a real gain smaller than the margin); neither -> inconclusive."""
    rows = []; a_any = False; b_by_g = {}
    for c in spec["freecands"]:
        ni = out[(out.kind == "NI") & (out.cand == c)]; a_ok = int((ni.holm < 0.05).sum()) >= 3; a_any |= a_ok
        rows.append({"candidate": c, "NI_holm_sig(of4)": int((ni.holm < 0.05).sum()), "A_noninferior": a_ok})
    sup_rows = []
    for g in [spec["reference"]] + spec.get("graphcands", []):
        ok = True
        for c in spec["freecands"]:
            sup = out[(out.kind == "SUP") & (out.graph == g) & (out.cand == c)]; c_ok = bool((sup.gain > 0).all() and int((sup.holm < 0.05).sum()) >= 3); ok &= c_ok
            sup_rows.append({"graph learner": g, "vs graph-free": c, "positive(of4)": int((sup.gain > 0).sum()), "Holm<0.05(of4)": int((sup.holm < 0.05).sum()), "beats it (B)": c_ok})
        b_by_g[g] = ok
    b_any = any(b_by_g.values())
    label = ("A and B (a real graph gain, smaller than the margin)" if a_any and b_any else "A: the graph is not needed (a graph-free candidate is non-inferior)" if a_any else "B: the graph adds (" + ", ".join(g for g, v in b_by_g.items() if v) + " beats every graph-free candidate)" if b_any else "inconclusive")
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(sup_rows)], ignore_index=True), label


if __name__ == "__main__":
    mode = sys.argv[1]; dirs = [a for a in sys.argv[2:] if not a.startswith("--") and not a.endswith(".json")]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    frame = load(dirs)
    if mode == "dev":
        for ref in (opts.get("ref", "lap3_k10_st4"), opts.get("ref2", "aw8_all_st2lr0.5")):
            print(f"\n#### learner - {ref} (positive = better than {ref}; seeds x budgets paired; draws pooled)\n"); t = dev_table(frame, ref); print(fmt_md(short(t)))
    elif mode == "budgets":
        ref = opts.get("ref", "lap3_k10_st4"); rows = []
        for split in sorted(frame.split.unique()):
            for l in sorted(set(frame.learner) - {ref}):
                for b in sorted(frame.budget.unique()):
                    g = frame[(frame.split == split) & (frame.budget == b)]; ga = g[g.learner == l].groupby("seed")[["auc", "acc"]].mean(); gb = g[g.learner == ref].groupby("seed")[["auc", "acc"]].mean()
                    d = (ga - gb).dropna(); rows.append({"split": split, "learner": l, "budget": b, "auc": d.auc.mean(), "acc": d.acc.mean()})
        print(fmt_md(pd.DataFrame(rows)))
    elif mode == "confirm":
        spec = json.loads(Path(sys.argv[2]).read_text()); out = run_tests(frame, spec); v, label = verdict(out, spec)
        cols = ["family", "split", "metric", "a", "b", "n_seeds", "gain", "lo", "hi", "share_better", "p", "holm"]
        print(fmt_md(out[cols])); print("\n" + fmt_md(v)); print(f"\nVERDICT: {label}")

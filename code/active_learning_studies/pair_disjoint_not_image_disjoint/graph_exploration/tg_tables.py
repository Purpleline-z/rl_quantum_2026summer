"""Tables of the temporal-graph study, generated from the result JSONs (nothing typed by hand).

  python3 tg_tables.py dev DIR [--base=lap3_k10_st4] [--metrics=auc,acc,far_auc,far_acc,ll] [--splits=A,B] [--learners=a,b]   paired gains of every learner over the baseline, per split
  python3 tg_tables.py confirm SPEC.json DIR [BLOCKED_DIR]                                                                   frozen test family: Holm over candidates x splits x {auc, acc}; verdicts; secondary endpoints
  python3 tg_tables.py budget DIR --cand=NAME [--base=...]                                                                    gain by budget
  python3 tg_tables.py audit DIR [--cand=lap3_k10_st4] [--ref=aw8_all_st2lr0.5]                                               gain by time-gap / anchor stratum from the per-judgment CSVs of tg_audit_time.py
  python3 tg_tables.py edges DIR                                                                                              sizes of the edge sets (mean over cells)

Gain = candidate - reference for auc / acc / far_* / near_* / inb_* ; reference - candidate for ll (positive = better).  Per-seed mean over (budget, draw) first, then Wilcoxon signed-rank over seeds (two-sided) and a
paired-bootstrap 95 % interval over seeds.  Holm is applied over the family stated in the spec.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SIGN = {"ll": -1, "far_ll": -1, "near_ll": -1, "inb_ll": -1, "htr_ll": -1}


def load(d) -> pd.DataFrame:
    """All records of a result directory (cells are written after every budget, so a seed enters a comparison only if both learners have every budget x draw of it: see ``per_seed_gain``)."""
    rows = [r for f in sorted(glob.glob(str(Path(d) / "seed*.json"))) for r in json.loads(Path(f).read_text())]; return pd.DataFrame(rows)


def per_seed_gain(df, cand, ref, metric, split) -> pd.Series:
    g = df[df.split == split]; a = g[g.learner == cand].set_index(["seed", "budget", "draw"]); b = g[g.learner == ref].set_index(["seed", "budget", "draw"]); common = a.index.intersection(b.index)
    if not len(common) or metric not in a: return pd.Series(dtype=float)
    need = df.budget.nunique() * (int(df.draw.max()) + 1); full = pd.Series(1, index=common).groupby(level=0).sum(); keep = full[full == need].index; common = [c for c in common if c[0] in set(keep)]
    if not len(common): return pd.Series(dtype=float)
    return (SIGN.get(metric, 1) * (a.loc[common, metric] - b.loc[common, metric])).groupby(level=0).mean().dropna()


def summarize(per: pd.Series) -> dict:
    if len(per) < 2: return {"gain": float("nan"), "lo": float("nan"), "hi": float("nan"), "share+": float("nan"), "p": float("nan"), "n": len(per)}
    bs = np.random.default_rng(0).choice(per.to_numpy(), (2000, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue)
    except ValueError: p = 1.0
    return {"gain": float(per.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "share+": float((per > 0).mean()), "p": p, "n": len(per)}


def holm(ps: dict) -> dict:
    """Holm step-down adjusted p-values for a dict key -> p."""
    keys = sorted(ps, key=lambda k: ps[k]); m = len(keys); adj = {}; run = 0.0
    for i, k in enumerate(keys): run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def md(df: pd.DataFrame, fmt="{:+.4f}") -> str:
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def dev_table(df, base, metrics, splits, learners=None) -> pd.DataFrame:
    rows = []
    for name in (learners or sorted(set(df.learner) - {base})):
        for split in splits:
            row = {"learner": name, "split": split}; n = 0
            for m in metrics:
                s = summarize(per_seed_gain(df, name, base, m, split)); row[m] = s["gain"]; row[m + "_p"] = s["p"]; n = max(n, s["n"])
            row["n"] = n; rows.append(row)
    return pd.DataFrame(rows)


def confirm(spec, df, blocked=None):
    base = spec["baseline"]; splits = spec.get("splits", ["A", "B"]); metrics = spec.get("metrics", ["auc", "acc"]); cands = spec["candidates"]; out = []
    tests = {(c["name"], s, m): summarize(per_seed_gain(df, c["name"], base, m, s)) for c in cands for s in splits for m in metrics}
    adj = holm({k: v["p"] for k, v in tests.items()})
    out.append(f"#### Main family: candidate - `{base}` (Holm over {len(tests)} tests = {len(cands)} candidates x {len(splits)} splits x {len(metrics)} metrics)\n")
    rows = [{"candidate": k[0], "split": k[1], "metric": k[2], "gain": v["gain"], "CI95": f"[{v['lo']:+.4f}, {v['hi']:+.4f}]", "share_seeds+": v["share+"], "p": v["p"], "p_holm": adj[k], "n_seeds": v["n"]} for k, v in tests.items()]
    out.append(md(pd.DataFrame(rows), "{:+.4f}").replace("+0.0000", "0")); verdicts = []
    for c in cands:
        name = c["name"]; mine = {k: v for k, v in tests.items() if k[0] == name}; pos = all(v["gain"] > 0 for v in mine.values()); nsig = sum(adj[k] < 0.05 for k in mine)
        ctrl_rows = []
        for ctrl in [x for x in (c.get("control"), c.get("control2")) if x]:
            cs = {(s, m): summarize(per_seed_gain(df, name, ctrl, m, s)) for s in splits for m in metrics}; ctrl_rows += [{"candidate": name, "control": ctrl, "split": s, "metric": m, "cand - control": v["gain"], "p (uncorrected)": v["p"]} for (s, m), v in cs.items()]
        beats = ctrl_rows and any(r["cand - control"] > 0 and r["p (uncorrected)"] < 0.05 for r in ctrl_rows if r["control"] == c.get("control"))
        verdicts.append({"candidate": name, "all 4 gains > 0": pos, "Holm-significant (of 4)": nsig, "beats shuffled control (>0 and p<.05 in >=1 cell)": bool(beats), "SUCCESS": bool(pos and nsig >= 3 and beats)})
        out.append(f"\n#### Controls of `{name}` (uncorrected)\n"); out.append(md(pd.DataFrame(ctrl_rows)))
    out.append("\n#### Verdicts (success rule: all four gains positive, >= 3 Holm-significant, better than the shuffled-time control)\n"); out.append(md(pd.DataFrame(verdicts)))
    sec = spec.get("secondary_metrics", ["far_auc", "far_acc", "near_auc", "near_acc", "ll", "htr_acc"])
    rows = []
    for c in cands:
        for s in splits:
            row = {"candidate": c["name"], "split": s}
            for m in sec: sm = summarize(per_seed_gain(df, c["name"], base, m, s)); row[m] = sm["gain"]; row[m + "_p"] = sm["p"]
            rows.append(row)
    out.append("\n#### Secondary endpoints (anchor-far = both test images > 0.03 from every typed anchor; anchor-near = complement; uncorrected p)\n"); out.append(md(pd.DataFrame(rows)))
    if blocked is not None and len(blocked):
        rows = []
        for c in cands:
            row = {"candidate": c["name"], "split": "blocked"}
            for m in spec.get("blocked_metrics", ["auc", "acc", "far_auc", "far_acc", "inb_auc", "inb_acc", "ll"]):
                sm = summarize(per_seed_gain(blocked, c["name"], base, m, "blocked")); row[m] = sm["gain"]; row[m + "_p"] = sm["p"]
            row["n"] = int(blocked[blocked.learner == c["name"]].seed.nunique()); rows.append(row)
        out.append("\n#### Secondary: red-team time-blocked split (one context per seed; `inb_*` = test judgments with both images inside a held-out block; uncorrected p)\n"); out.append(md(pd.DataFrame(rows)))
    return "\n".join(out)


def budget_table(df, cand, base, metrics=("auc", "acc", "far_auc", "far_acc")) -> pd.DataFrame:
    rows = []
    for split in sorted(set(df.split)):
        for b in sorted(set(df.budget)):
            sub = df[df.budget == b]; row = {"split": split, "budget": b}
            for m in metrics: row[m] = summarize(per_seed_gain(sub, cand, base, m, split))["gain"]
            rows.append(row)
    return pd.DataFrame(rows)


def audit_tables(d, cand, ref, bins=(5, 15, 40)) -> str:
    """Per-judgment CSVs -> (1) overall per-seed gains (auc, acc, anchor-far auc / acc) and (2) accuracy / log-loss gain by time gap to the nearest labelled frame and by anchor proximity."""
    from sklearn.metrics import roc_auc_score
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(d) / "seed*.csv.gz")))], ignore_index=True); keys = ["seed", "split", "budget", "draw", "j"]
    a = df[df.learner == cand].set_index(keys); b = df[df.learner == ref].set_index(keys)[["d"]]; j = a.join(b, rsuffix="_ref", how="inner"); z = j.d * j.sign; zr = j.d_ref * j.sign
    j["acc_c"], j["acc_r"] = (z > 0).astype(float), (zr > 0).astype(float); j["ll_c"], j["ll_r"] = np.logaddexp(0, -z), np.logaddexp(0, -zr); j = j.reset_index(); j["far"] = j.anc_all_min > 0.03
    out = [f"#### Overall and anchor-far (per-seed means over budgets; Wilcoxon over seeds): `{cand}` - `{ref}`\n"]; rows = []
    for split, g in j.groupby("split"):
        row = {"split": split}
        for tag, sub in (("all", g), ("far", g[g.far])):
            per = {"auc": [], "acc": []}; seeds = []
            for (seed, budget, draw), c in sub.groupby(["seed", "budget", "draw"]):
                if c.sign.nunique() < 2: continue
                per["auc"].append((seed, roc_auc_score(c.sign > 0, c.d) - roc_auc_score(c.sign > 0, c.d_ref))); per["acc"].append((seed, (c.weight * (c.acc_c - c.acc_r)).sum() / c.weight.sum()))
            for m in ("auc", "acc"):
                s_ = pd.DataFrame(per[m], columns=["seed", "g"]).groupby("seed").g.mean(); sm = summarize(s_); row[f"{tag}_{m}"] = sm["gain"]; row[f"{tag}_{m}_p"] = sm["p"]; row["n_seeds"] = sm["n"]
        rows.append(row)
    out.append(md(pd.DataFrame(rows))); edges = [-1, *bins, 1e9]; labels = [f"<= {bins[0]}"] + [f"{bins[i]+1}-{bins[i+1]}" for i in range(len(bins) - 1)] + [f"> {bins[-1]}"]
    j["time_gap"] = pd.cut(j.tgap_lab, edges, labels=labels); out.append("\n#### By frame gap of the test judgment's farther image to the nearest labelled frame of the same run (accuracy gain weighted; log-loss gain = ref - cand)\n"); rows = []
    for (split, st), g in j.groupby(["split", "time_gap"], observed=True):
        per = g.groupby("seed").apply(lambda x: pd.Series({"acc_gain": (x.weight * (x.acc_c - x.acc_r)).sum() / x.weight.sum(), "ll_gain": (x.weight * (x.ll_r - x.ll_c)).sum() / x.weight.sum(), "acc_ref": (x.weight * x.acc_r).sum() / x.weight.sum()}))
        sa, sl = summarize(per.acc_gain), summarize(per.ll_gain); rows.append({"split": split, "frame gap": str(st), "n_judgments": len(g), "share": len(g) / len(j[j.split == split]), "acc_ref": per.acc_ref.mean(), "acc_gain": sa["gain"], "acc_p": sa["p"], "ll_gain": sl["gain"], "ll_p": sl["p"]})
    out.append(md(pd.DataFrame(rows)))
    j["anchor"] = np.where(j.far, "far (>0.03)", "near (<=0.03)"); out.append("\n#### By anchor proximity x frame gap\n"); rows = []
    for (split, an, st), g in j.groupby(["split", "anchor", "time_gap"], observed=True):
        per = g.groupby("seed").apply(lambda x: pd.Series({"acc_gain": (x.weight * (x.acc_c - x.acc_r)).sum() / x.weight.sum(), "acc_ref": (x.weight * x.acc_r).sum() / x.weight.sum()})); sa = summarize(per.acc_gain)
        rows.append({"split": split, "anchor": an, "frame gap": str(st), "n_judgments": len(g), "acc_ref": per.acc_ref.mean(), "acc_gain": sa["gain"], "acc_p": sa["p"]})
    out.append(md(pd.DataFrame(rows))); return "\n".join(out)


if __name__ == "__main__":
    mode = sys.argv[1]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a); args = [a for a in sys.argv[2:] if not a.startswith("--")]
    if mode == "dev":
        df = load(args[0]); t = dev_table(df, opts.get("base", "lap3_k10_st4"), opts.get("metrics", "auc,acc,far_auc,far_acc,ll").split(","), opts.get("splits", "A,B").split(","), opts["learners"].split(",") if "learners" in opts else None); print(md(t))
    elif mode == "confirm":
        spec = json.loads(Path(args[0]).read_text()); df = load(args[1]); blocked = load(args[2]) if len(args) > 2 else None; print(confirm(spec, df, blocked))
    elif mode == "budget":
        df = load(args[0]); print(md(budget_table(df, opts["cand"], opts.get("base", "lap3_k10_st4"))))
    elif mode == "audit":
        print(audit_tables(args[0], opts.get("cand", "lap3_k10_st4"), opts.get("ref", "aw8_all_st2lr0.5")))
    elif mode == "edges":
        df = load(args[0]); print(md(df[df.learner.str.startswith("tg_")].groupby(["learner", "split"])[["n_knn", "n_temporal"]].mean().reset_index(), "{:.0f}"))

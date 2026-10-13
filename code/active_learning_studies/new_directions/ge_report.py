"""Generated tables and verdicts for the graph-endpoint study (PREREGISTRATION_GE1).  python3 ge_report.py --out TABLES.md [--template T --doc D]."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import ge_analyze as ga

R = ga.R
CAND_K = [("laplr", "inductive", -1), ("laplr", "transductive", -1), ("sgc", "inductive", -1), ("sgc", "transductive", -1), ("lap3_ref", "inductive", 60)]
REFS = {"nn": ("nn", "none", -1), "logreg": ("logreg", "none", -1)}
label = lambda k: f"{k[0]} ({k[1]}{', ' + str(k[2]) + ' judgments' if k[2] > 0 else ''})"


# ------------------------------------------------------------------------------------------ (a) classification
def class_family(frame: pd.DataFrame, ref_key) -> pd.DataFrame:
    per = frame.groupby(["seed", "method", "protocol", "budget"], as_index=False).acc.mean(); get = lambda k: per[(per.method == k[0]) & (per.protocol == k[1]) & (per.budget == k[2])].set_index("seed").acc
    base = get(ref_key); rows = []
    for k in CAND_K:
        a = get(k); d = (a - base).dropna(); rows.append({"method": label(k), "n_seeds": len(d), "acc": float(a.mean()), "ref_acc": float(base.mean()), "diff": float(d.mean()), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "sd_diff": float(d.std(ddof=1)), "mde80": float(2.8 * d.std(ddof=1) / np.sqrt(len(d))), "p": ga.wil(d)})
    t = pd.DataFrame(rows); hp = ga.holm(dict(enumerate(t.p))); t["holm"] = [hp[i] for i in range(len(t))]; return t


def class_controls(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [(("lap3_ref", "inductive", 60), ("aw8_ref", "none", 60), "lap3_ref - aw8_ref (graph vs non-graph head)"), (("lap3_ref", "inductive", 60), ("lap3_ref_shuf", "inductive", 60), "lap3_ref - shuffled graph"),
             (("laplr", "inductive", -1), ("laplr_shuf", "inductive", -1), "laplr - shuffled graph"), (("laplr", "inductive", -1), REFS["logreg"], "laplr - logreg (gamma=0 limit)"), (("sgc", "transductive", -1), ("sgc", "inductive", -1), "sgc transductive - inductive")]
    rows = []
    for a, b, name in pairs:
        r = ga.class_vs(frame, a, b); rows.append({"comparison": name, "n_seeds": r["n"], "diff": r["diff"], "p_uncorrected": r["p"]})
    return pd.DataFrame(rows)


def class_block(name: str, title: str):
    path = R / f"{name}_class"
    if not path.exists() or not list(path.glob("seed*.json")): return f"### {title}\n\n(not run yet)\n\n", None
    f = ga.load(path); k1 = class_family(f, REFS["nn"]); k2 = class_family(f, REFS["logreg"]); refs = f[f.budget == -1].groupby("method").acc.mean()
    text = f"### {title}\n\nseeds: {f.seed.nunique()}; mean accuracy of the controls: 1-NN {refs.get('nn', float('nan')):.3f}, logistic regression {refs.get('logreg', float('nan')):.3f}.\n\n**K1: candidates minus 1-NN (Holm over 5)**\n\n" + ga.md(k1) + "\n**K2 (secondary): candidates minus logistic regression (Holm over 5)**\n\n" + ga.md(k2) + "\n**Controls (uncorrected)**\n\n" + ga.md(class_controls(f)) + "\n"
    return text, {"k1": k1, "k2": k2}


# ------------------------------------------------------------------------------------------ (b) dose-response
def dose_block(name: str, title: str):
    path = R / f"{name}_dose"
    if not path.exists() or not list(path.glob("seed*_*.json")): return f"### {title}\n\n(not run yet)\n\n", None
    f = ga.load(path, "seed*_*.json"); g = ga.dose_gains(f); tr = pd.concat([ga.dose_trend(g, "all", "auc"), ga.dose_trend(g, "all", "acc")]); hp = ga.holm(dict(enumerate(tr.p))); tr["holm"] = [hp[i] for i in range(len(tr))]
    out = ga.dose_trend(g, "out", "auc"), ga.dose_trend(g, "out", "acc"); ex = pd.concat([ga.dose_trend(g, "all", m, drop_zero=True) for m in ("auc", "acc")]); levels = ga.dose_table(g)
    plateau = []
    for split, gg in g.groupby("split"):
        piv = {m: gg.pivot(index="seed", columns="variant", values=m) for m in ("auc", "acc")}
        for m in ("auc", "acc"):
            d = (piv[m]["all_p100"] - piv[m]["all_p25"]).dropna(); plateau.append({"split": split, "metric": m, "n_seeds": len(d), "diff_p100_minus_p25": float(d.mean()), "p_uncorrected": ga.wil(d)})
    text = f"### {title}\n\nseeds: {f.seed.nunique()}; gains = `lap3_k10` with the given node set minus `aw8_all` (mean over budgets and draws).\n\n**Gain at every level**\n\n" + ga.md(levels) + "\n**Pre-registered trend tests, series 'all' (Holm over 4)**\n\n" + ga.md(tr) + "\n**Plateau check: gain(100%) − gain(25%) (uncorrected)**\n\n" + ga.md(pd.DataFrame(plateau)) + "\n**Exploratory: trend over the levels 10%, 25%, 50%, 100% only (p = 0 dropped; uncorrected)**\n\n" + ga.md(ex) + "\n**Series 'out' (descriptive)**\n\n" + ga.md(pd.concat(out)) + "\n"
    return text, {"trend": tr}


# ------------------------------------------------------------------------------------------ (c) harsh splits
def bootstrap_gain(path: Path, kind: str, arm: str, ref: str, reps: int = 300, seed: int = 0) -> dict:
    import json as _json
    cache_file = R / "bootstrap_cache.json"; cache = _json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = f"{path.name}|{kind}|{arm}|{ref}|{reps}|{seed}|{len(list(path.glob(f'seed*_{kind}.d.npz')))}"
    if key in cache: return {k: (tuple(v) if isinstance(v, list) else v) for k, v in cache[key].items()}
    out = _bootstrap_gain(path, kind, arm, ref, reps, seed)
    if out: cache[key] = out; cache_file.write_text(_json.dumps(cache))
    return out


def _bootstrap_gain(path: Path, kind: str, arm: str, ref: str, reps: int = 300, seed: int = 0) -> dict:
    """Test-group bootstrap of the mean gain (arm - ref) over seeds, budgets and draws: groups of the (fixed) test set are resampled with replacement, the same resample for every seed."""
    files = sorted(path.glob(f"seed*_{kind}.d.npz"))
    if not files: return {}
    z0 = np.load(files[0], allow_pickle=True); groups = z0["meta_group"]; uniq = np.unique(groups); rng = np.random.default_rng(seed); index = {g: np.flatnonzero(groups == g) for g in uniq}; data = []
    for f in files:
        z = np.load(f, allow_pickle=True); keys = sorted({k[2:].rsplit("_", 1)[0] for k in z.files if k.startswith("d_")})
        for key in keys:
            if f"d_{key}_{arm}" in z.files and f"d_{key}_{ref}" in z.files: data.append((z[f"d_{key}_{arm}"], z[f"d_{key}_{ref}"], z["meta_sign"], z["meta_weight"]))
    def metrics(d, s, w, idx):
        d, s, w = d[idx], s[idx], w[idx]
        return (roc_auc_score(s > 0, d) if len(set(s > 0)) == 2 else np.nan), float((w * ((d * s) > 0)).sum() / w.sum())
    full = np.array([[*(np.subtract(metrics(a, s, w, np.arange(len(s))), metrics(b, s, w, np.arange(len(s)))))] for a, b, s, w in data]).mean(0); boots = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq)); idx = np.concatenate([index[g] for g in pick]); boots.append(np.nanmean([np.subtract(metrics(a, s, w, idx), metrics(b, s, w, idx)) for a, b, s, w in data], axis=0))
    boots = np.array(boots); return {"auc": (float(full[0]), *map(float, np.nanpercentile(boots[:, 0], [2.5, 97.5]))), "acc": (float(full[1]), *map(float, np.nanpercentile(boots[:, 1], [2.5, 97.5]))), "n_groups": len(uniq), "n_judgments": len(groups)}


def harsh_block(name: str, title: str, bootstrap: bool = True):
    path = R / f"{name}_harsh"
    if not path.exists() or not list(path.glob("seed*_*.json")): return f"### {title}\n\n(not run yet)\n\n", None
    f = ga.load(path, "seed*_*.json"); gains = ga.arm_gains(f, ["kind"], "arm", "base"); fam = gains[gains.arm == "lap3"].copy(); hp = ga.holm(dict(enumerate(fam.p))); fam["holm"] = [hp[i] for i in range(len(fam))]
    shuf = ga.arm_vs(f, ["kind"], "arm", "lap3", "lap3_shuf"); noisy = ga.arm_vs(f, ["kind"], "arm", "lap3", "lap3_n3")
    text = f"### {title}\n\nseeds: {f.seed.nunique()}; test judgments pooled over the folds of the split kind (fixed across seeds).\n\n**Pre-registered family H: `lap3_k10` − `aw8_all` (Holm over 4)**\n\n" + ga.md(fam) + "\n**All arms vs `aw8_all` (descriptive)**\n\n" + ga.md(gains) + "\n**`lap3_k10` − shuffled-graph control (uncorrected)**\n\n" + ga.md(shuf) + "\n**`lap3_k10` − noisy-graph control (uncorrected)**\n\n" + ga.md(noisy) + "\n"
    if bootstrap:
        rows = []
        for kind in ("run", "temp"):
            b = bootstrap_gain(path, kind, "lap3", "base")
            if b: rows += [{"kind": kind, "metric": m, "mean_gain": b[m][0], "ci_lo": b[m][1], "ci_hi": b[m][2], "test_groups": b["n_groups"], "test_judgments": b["n_judgments"]} for m in ("auc", "acc")]
        if rows: text += "**Test-group bootstrap of the `lap3_k10` − `aw8_all` gain (300 resamples of the fixed test groups; caveat, not part of the criterion)**\n\n" + ga.md(pd.DataFrame(rows)) + "\n"
    return text, {"fam": fam, "shuf": shuf}


# ------------------------------------------------------------------------------------------ verdicts
def verdict_a(res) -> str:
    if not res: return "not run yet"
    out = []
    for proto in ("inductive", "transductive"):
        k = res["k1"][res["k1"].method.str.contains(proto)]; ok = k[(k["diff"] > 0) & (k.holm < .05)]
        out.append(f"{proto}: {'MET by ' + ', '.join(ok.method) if len(ok) else 'not met'} (best diff vs 1-NN {k['diff'].max():+.3f}, smallest Holm {k.holm.min():.3f})")
    return "; ".join(out)


def verdict_b(res) -> str:
    if not res: return "not run yet"
    t = res["trend"]; n = int(((t.slope > 0) & (t.holm < .05)).sum()); return f"{n} of {len(t)} trend tests positive with Holm < 0.05 -> {'MET' if n >= 3 else 'not met'}"


def verdict_c(res) -> str:
    if not res: return "not run yet"
    fam, shuf = res["fam"], res["shuf"]; pos = int((fam["gain"] > 0).sum()); sig = int(((fam["gain"] > 0) & (fam.holm < .05)).sum())
    beats = all(((shuf[(shuf.kind == k)]["diff"] > 0) & (shuf[(shuf.kind == k)].p_uncorrected < .05)).any() for k in ("run", "temp"))
    return f"{pos} of 4 gains positive, {sig} with Holm < 0.05, beats shuffled control in both kinds: {beats} -> {'MET' if pos == 4 and sig >= 3 and beats else 'not met'}"


def dev_tables() -> str:
    out = ""
    f = pd.concat([ga.load(R / d) for d in ("dev_class", "dev_class_ideal", "dev_class_laplr") if (R / d).exists()])
    if len(f): out += "### Dev: classification endpoint (seeds 8400–8419; NOT evidence)\n\n" + ga.md(ga.class_table(f).drop(columns=["n_seeds"])) + "\n"
    if (R / "dev_dose").exists():
        g = ga.dose_gains(ga.load(R / "dev_dose", "seed*_*.json")); out += "### Dev: dose-response (NOT evidence)\n\n" + ga.md(ga.dose_table(g)) + "\n" + ga.md(pd.concat([ga.dose_trend(g, s, m) for s in ("all", "out") for m in ("auc", "acc")])) + "\n"
    if (R / "dev_harsh").exists():
        f = ga.load(R / "dev_harsh", "seed*_*.json"); out += "### Dev: harsh splits (NOT evidence)\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n"
    if (R / "dev_lam").exists():
        f = ga.load(R / "dev_lam", "seed*_*.json"); out += "### Dev: lambda selected by internal validation (NOT evidence)\n\n" + ga.md(ga.arm_gains(f, ["split"], "arm", "base")) + "\n" + ga.md(ga.arm_vs(f, ["split"], "arm", "lap_auto", "lap3")) + "\n"
        a = f[f.arm == "lap_auto"].chosen_lambda.value_counts(normalize=True).round(3).to_dict(); out += f"Chosen lambda shares: {a}\n\n"
    return out


def exploratory_tables(bootstrap: bool = True) -> str:
    out = ""
    if (R / "dev_nodes").exists() and list((R / "dev_nodes").glob("seed*_*.json")):
        f = ga.load(R / "dev_nodes", "seed*_*.json"); t = ga.arm_gains(f, ["split"], "variant", "base"); t["n_nodes"] = [f[(f.variant == a) & (f.split == s)].n_nodes.mean() for a, s in zip(t.arm, t.split)]
        out += "### Which graph nodes carry the gain? (lap3_k10 − aw8_all, preference endpoint, standard splits; node sets: ideal = ideal images only; train = ideal + images of the labelled judgments; cand = ideal + unlabelled candidate-pool images; groups = ideal + all pair-group images; all = standard graph)\n\n" + ga.md(t[["split", "arm", "metric", "n_nodes", "gain", "share_better", "p"]]) + "\n"
    if (R / "dev_lamclass").exists():
        f = ga.load(R / "dev_lamclass"); g = f.groupby(["seed", "lam"], as_index=False).acc.mean(); piv = g.pivot(index="seed", columns="lam", values="acc")
        rows = [{"lambda": l, "n_seeds": len(piv), "acc": float(piv[l].mean()), "diff_vs_lambda0": float((piv[l] - piv[0.0]).mean()), "p_uncorrected": ga.wil(piv[l] - piv[0.0])} for l in piv.columns]
        out += "### Does the Laplacian term hurt type classification? (reference-only anchors, 60 judgments, inductive graph, dev seeds)\n\n" + ga.md(pd.DataFrame(rows)) + "\n"
    if (R / "dev_temp3").exists() and list((R / "dev_temp3").glob("seed*_*.json")):
        f = ga.load(R / "dev_temp3", "seed*_*.json"); out += "### Three temperature terciles held out in turn (dev seeds; more test groups than the top tercile alone)\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n" + ga.md(ga.arm_vs(f, ["kind"], "arm", "lap3", "lap3_shuf")) + "\n"
        if bootstrap:
            b = bootstrap_gain(R / "dev_temp3", "temp3", "lap3", "base")
            if b: out += "Test-group bootstrap of the lap3 − aw8_all gain (all three bands pooled): " + "; ".join(f"{m}: {b[m][0]:+.3f} [{b[m][1]:+.3f}, {b[m][2]:+.3f}]" for m in ("auc", "acc")) + f" ({b['n_groups']} test groups, {b['n_judgments']} judgments)\n\n"
    if (R / "dev_trans").exists() and list((R / "dev_trans").glob("seed*_*.json")):
        f = ga.load(R / "dev_trans", "seed*_*.json"); out += "### Unlabelled frames of the held-out condition in the graph (lap3_T: transductive, held-out frames and test images are unlabelled nodes) vs inductive lap3 (dev seeds)\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n" + ga.md(ga.arm_vs(f, ["kind"], "arm", "lap3_T", "lap3")) + "\n"
    if (R / "dev_target").exists() and list((R / "dev_target").glob("seed*_*.json")):
        f = ga.load(R / "dev_target", "seed*_*.json"); out += "### Dose-response of the held-out condition's UNLABELLED frames (lapF25/50/100 = that fraction of the held-out run/band frames, test images excluded, enters the graph; lap3_T = all frames plus the test images; dev seeds)\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n"
        out += ga.md(pd.concat([ga.arm_vs(f, ["kind"], "arm", a, "lap3") for a in ("lapF25", "lapF50", "lapF100", "lap3_T")])) + "\n"
    rows = []
    for b in ("conf1", "conf2"):
        if not (R / f"{b}_dose").exists(): continue
        f = ga.load(R / f"{b}_dose", "seed*_*.json"); per = f.groupby(["split", "seed", "variant"], as_index=False)[["htr_acc", "acc"]].mean()
        for split, g in per.groupby("split"):
            piv = g.pivot(index="seed", columns="variant", values="htr_acc"); d = piv["all_p100"] - piv["base"]; rows.append({"block": b, "split": split, "n_seeds": len(d), "htr_acc_gain": float(d.mean()), "p_uncorrected": ga.wil(d)})
    if rows: out += "### HTR accuracy gain of lap3_k10 over aw8_all on the ORDINARY splits, from the confirmation dose blocks (descriptive, not pre-registered; HTR is the lab's priority type)\n\n" + ga.md(pd.DataFrame(rows)) + "\n"
    rows = []
    for b in ("conf1", "conf2"):
        if not (R / f"{b}_class").exists(): continue
        f = ga.load(R / f"{b}_class"); per = f.groupby(["seed", "method"], as_index=False)[[f"recall_{j}" for j in range(4)]].mean(); a = per[per.method == "lap3_ref"].set_index("seed"); ref = per[per.method == "aw8_ref"].set_index("seed")
        for j, name in enumerate(("(1x1)", "c(6x2)", "(sqrt13)", "HTR")): d = (a[f"recall_{j}"] - ref[f"recall_{j}"]).dropna(); rows.append({"block": b, "class": name, "recall_change_lap3_ref_minus_aw8_ref": float(d.mean()), "p_uncorrected": ga.wil(d)})
    if rows: out += "### Which classes does the Laplacian head lose? Recall change of lap3_ref relative to aw8_ref (classification endpoint, 60 judgments; descriptive)\n\n" + ga.md(pd.DataFrame(rows)) + "\n"
    if (R / "dev_traj").exists() and list((R / "dev_traj").glob("seed*.json")):
        f = ga.load(R / "dev_traj"); c = f[f.endpoint == "class"]; p = f[f.endpoint == "pref"]
        out += "### Trajectory-only graph (no ideal-image nodes) vs the full graph (dev seeds): classification accuracy (lap3 variants minus aw8_ref) and preference endpoint (minus aw8_all)\n\n" + ga.md(ga.arm_gains(c, ["endpoint"], "arm", "aw8_ref", metrics=("acc",))) + "\n" + ga.md(ga.arm_gains(p, ["split"], "arm", "aw8_all")) + "\n"
    if (R / "dev_late").exists() and list((R / "dev_late").glob("seed*_*.json")):
        f = ga.load(R / "dev_late", "seed*_*.json"); out += "### Forward-in-time split: the last third (by frame index) of every growth run held out (dev seeds; 25 test groups, 36 decisive judgments)\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n" + ga.md(ga.arm_vs(f, ["kind"], "arm", "lap3", "lap3_shuf")) + "\n"
        if bootstrap:
            b = bootstrap_gain(R / "dev_late", "late", "lap3", "base")
            if b: out += "Test-group bootstrap of the lap3 − aw8_all gain: " + "; ".join(f"{m}: {b[m][0]:+.3f} [{b[m][1]:+.3f}, {b[m][2]:+.3f}]" for m in ("auc", "acc")) + f" ({b['n_groups']} test groups, {b['n_judgments']} judgments)\n\n"
    if (R / "dev_ridge").exists() and list((R / "dev_ridge").glob("seed*_*.json")):
        f = ga.load(R / "dev_ridge", "seed*_*.json"); out += "### Generic score shrinkage without a graph (ridge1/ridge3 = shrink random nodes' scores towards 0) vs lap3 vs shuffled graph on the three harsh split kinds, minus aw8_all (dev seeds)\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n"
    if (R / "dev_cut").exists() and list((R / "dev_cut").glob("seed*.json")):
        f = ga.load(R / "dev_cut"); c = f[f.endpoint == "class"]; p = f[f.endpoint == "pref"]
        out += "### Class-aware edge cutting (lap3_cut: kNN edges between labelled ideal images of different classes removed) vs the full graph (dev seeds): classification accuracy (minus aw8_ref) and preference endpoint (minus aw8_all)\n\n" + ga.md(ga.arm_gains(c, ["endpoint"], "arm", "aw8_ref", metrics=("acc",))) + "\n" + ga.md(ga.arm_gains(p, ["split"], "arm", "aw8_all")) + "\n"
    if (R / "dev_k").exists() and list((R / "dev_k").glob("seed*_*.json")):
        f = ga.load(R / "dev_k", "seed*_*.json"); out += "### Graph degree on the harsh splits (dev seeds): lap3 = k 10 (frozen), lap3_k5, lap3_k20 minus aw8_all\n\n" + ga.md(ga.arm_gains(f, ["kind"], "arm", "base")) + "\n"
    import ge_htr
    try:
        t = ge_htr.table()
        if len(t): out += "### Per-type view of the harsh-split accuracy gain in the confirmation blocks (lap3_k10 − aw8_all; HTR vs the other types; test-group bootstrap; descriptive — only 7 HTR test judgments)\n\n" + ga.md(t) + "\n"
    except Exception as e: out += f"(per-type table unavailable: {e})\n"
    return out or "(no exploratory runs yet)\n"


def sections(bootstrap: bool = True) -> dict:
    sec = {}; res = {}
    for blk, tag in (("conf1", "Confirmation 1 (seeds 8500–8534)"), ("conf2", "Confirmation 2 (seeds 8550–8584, replication)")):
        for key, fn in (("class", class_block), ("dose", dose_block), ("harsh", harsh_block)):
            kw = {"bootstrap": bootstrap} if key == "harsh" else {}
            text, r = fn(blk, f"{tag}: " + {"class": "classification endpoint (28 outer-test ideal images)", "dose": "dose-response of unlabelled images", "harsh": "harsh preference splits"}[key], **kw); sec[f"{blk}_{key}"] = text; res[(blk, key)] = r
    sec["verdicts"] = "".join(f"- **{blk}**: (a) classification: {verdict_a(res[(blk, 'class')])}. (b) dose-response: {verdict_b(res[(blk, 'dose')])}. (c) harsh splits: {verdict_c(res[(blk, 'harsh')])}.\n" for blk in ("conf1", "conf2"))
    sec["dev"] = dev_tables(); sec["exploratory"] = exploratory_tables(bootstrap); return sec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--template", default=None); ap.add_argument("--doc", default=None); ap.add_argument("--no-bootstrap", action="store_true"); a = ap.parse_args(); sec = sections(not a.no_bootstrap)
    if a.out:
        text = "# Tables for the graph-endpoint study (generated by ge_report.py; do not edit by hand)\n\n## Pre-registered readings\n\n" + sec["verdicts"] + "\n" + "".join(sec[f"{b}_{k}"] for b in ("conf1", "conf2") for k in ("class", "dose", "harsh")) + "\n## Exploratory runs (not evidence)\n\n" + sec["exploratory"] + "\n## Development screens (not evidence)\n\n" + sec["dev"]
        Path(a.out).write_text(text); print(f"wrote {a.out} ({len(text)} chars)")
    if a.template and a.doc:
        t = Path(a.template).read_text()
        for k, v in sec.items(): t = t.replace(f"<!--TABLE:{k}-->", v).replace(f"<!--SECTION:{k}-->", v)
        Path(a.doc).write_text(t); print(f"wrote {a.doc}")


if __name__ == "__main__":
    main()

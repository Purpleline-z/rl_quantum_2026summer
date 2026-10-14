#!/usr/bin/env python3
"""Pre-registered tests of the transductive study (frozen spec TD1_spec.json; result directories under new_directions/results/td).

  python3 td_confirm.py TD1_spec.json [--blocks confirm1,confirm2] [--boot 1000]

Family (Holm over all of it): for every candidate, forward windows FA and FB x {AUC, accuracy}, per-seed mean over the six cuts of (candidate - BASE); two-sided Wilcoxon over seeds.
Success of a candidate: >= 3 of its 4 tests positive with Holm p < 0.05 AND it beats its shuffled-graph control (AUC gain over the shuffled control positive with uncorrected p < 0.05 in both windows).
Secondary (descriptive, own Holm family): standard random splits A / B (budgets pooled) vs the inductive BASE; time-blocked split; anchor-far stratum; decomposition BASE -> STD -> trans_test -> trans_win -> trans_fut.
Intervals: seed bootstrap and test-group bootstrap (the group-level interval treats the 168 labelled groups as the sampling units).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import td_tables as T

HERE = Path(__file__).resolve().parent


def verdicts(spec: dict, fam: pd.DataFrame, shuf: dict, kinds=("global", "targeted")) -> list:
    out = []
    for c in spec["candidates"]:
        sub = fam[fam.learner == c["learner"]]; wins = int(((sub.gain > 0) & (sub.holm_p < 0.05)).sum())
        beats = all(shuf.get((c["learner"], w, k), (-1, 1))[0] > 0 and shuf.get((c["learner"], w, k), (-1, 1))[1] < 0.05 for w in spec["windows"] for k in kinds)
        out.append(f"- **{c['id']} `{c['learner']}`**: {wins} of {len(sub)} tests positive with Holm p < 0.05; beats its shuffled controls ({'+'.join(kinds)}; AUC, both windows, uncorrected p < 0.05): {beats} -> " + ("**SUCCESS**" if wins >= spec["min_wins"] and beats else "not met"))
    return out


def forward_tables(spec: dict, dirs, boot: int) -> str:
    df, meta = T.load_with_meta(dirs); base = spec["baseline"]; cands = [c["learner"] for c in spec["candidates"]]; text = []
    n = df[(df.learner == base)].groupby(["window"]).n_test.agg(["mean", "min", "max"]); nj = df[(df.learner == base)].groupby(["seed", "cut", "window"]).n_test.first().groupby("window").sum()
    text.append(f"{df.seed.nunique()} seeds x 6 cuts; decisive test judgments per (seed, cut, window): " + "; ".join(f"{w} mean {r['mean']:.0f} (min {r['min']:.0f}, max {r['max']:.0f})" for w, r in n.iterrows()) + "; distinct test groups per window pooled over cuts and seeds are the groups of the 168 (windows overlap between cuts).")
    fam = T.test_family(df, meta, base, cands, spec["windows"], tuple(spec["metrics"]), boot); text += ["\n**Primary family (forward endpoint), gain over BASE = `lap3_k10_st4` on the past-only graph**\n", T.md(fam)]
    shuf = {}; rows = []; have = set(df.learner); kinds = ["global"] + (["targeted"] if all(c["tshuffled"] in have for c in spec["candidates"]) else [])
    for c in spec["candidates"]:
        for kind, key in (("global", "shuffled"), ("targeted", "tshuffled")):
            if kind not in kinds: continue
            for w in spec["windows"]:
                for m in ("auc", "acc"):
                    s = T.summarize(T.per_seed(df, c["learner"], c[key], w, m)); rows.append({"learner": c["learner"], "control": c[key], "window": w, "metric": m, "gain": s["gain"], "seed_ci": f"[{s['lo']:+.4f},{s['hi']:+.4f}]", "p": s["p"]})
                    if m == "auc": shuf[(c["learner"], w, kind)] = (s["gain"], s["p"])
    text += ["\n**Candidate minus its shuffled-graph controls** (global shuffle; targeted shuffle of the held-out nodes only, where run; uncorrected)\n", T.md(pd.DataFrame(rows))]
    text += ["\n**Verdicts**\n"] + verdicts(spec, fam, shuf, tuple(kinds))
    ref = []
    for name, lab in [(spec.get("tuned_graph_free"), "tuned graph-free control"), (spec.get("graph_free_control"), "graph-free transductive control (dropout consistency on the same nodes)")] + [(x, "decomposition") for x in spec.get("decomposition", [])]:
        if not name: continue
        for w in spec["windows"]:
            r = {"learner": name, "role": lab, "window": w}
            for m in ("auc", "acc", "ll", "far_auc"): s = T.summarize(T.per_seed(df, name, base, w, m)); r[m] = s["gain"]; r[m + "_p"] = s["p"]
            ref.append(r)
    text += ["\n**Reference learners minus BASE** (descriptive; log-loss: BASE - learner)\n", T.md(pd.DataFrame(ref))]
    far = []
    for c in cands:
        for w in spec["windows"]:
            r = {"learner": c, "window": w}
            for m in ("far_auc", "far_acc", "far_ll"): s = T.summarize(T.per_seed(df, c, base, w, m)); r[m] = s["gain"]; r[m + "_ci"] = f"[{s['lo']:+.3f},{s['hi']:+.3f}]"
            far.append(r)
    text += ["\n**Anchor-far stratum** (both images > 0.03 cosine distance from every typed anchor; per-(seed, cut) metric where defined; descriptive)\n", T.md(pd.DataFrame(far))]
    return "\n".join(text)


def standard_tables(spec: dict, dirs, kind: str, boot: int) -> str:
    df, meta = T.load_with_meta(dirs); base = spec["baseline"]
    std_c = list({c["standard"]: {"learner": c["standard"], "shuffled": c["standard_shuffled"], "ids": []} for c in spec["candidates"]}.values())   # in the standard split all transductive node sets coincide: distinct learners only
    for c in spec["candidates"]:
        for x in std_c:
            if x["learner"] == c["standard"]: x["ids"].append(c["id"])
    spec = {**spec, "candidates": std_c}; cands = [c["learner"] for c in std_c]
    fams = []
    splits = ["A", "B"] if kind == "standard" else ["blocked"]
    for sp in splits:
        d = df[df.split == sp].copy(); d["window"] = "T"; m = {k: v for k, v in meta.items() if k[1] == sp}
        t = T.test_family(d, m, base, cands, ["T"], ("auc", "acc"), boot); t.insert(1, "split", sp); fams.append(t)
    fam = pd.concat(fams, ignore_index=True); adj = T.holm({i: p for i, p in enumerate(fam.p)}); fam["holm_p"] = [adj[i] for i in range(len(fam))]
    out = [f"{df.seed.nunique()} seeds; test judgments per context: mean {df[df.learner == base].n_test.mean():.0f}\n", T.md(fam.drop(columns=["window"]))]
    rows = []
    for c in spec["candidates"]:
        for sp in splits:
            d = df[df.split == sp].copy(); d["window"] = "T"
            for m in ("auc", "acc"): s = T.summarize(T.per_seed(d, c["learner"], c["shuffled"], "T", m)); rows.append({"learner": c["learner"], "control": c["shuffled"], "split": sp, "metric": m, "gain": s["gain"], "p": s["p"]})
    out += ["\n**Candidate minus its shuffled control**\n", T.md(pd.DataFrame(rows))]
    ref = []
    for name in [spec.get("tuned_graph_free"), spec.get("graph_free_control_standard")]:
        if not name: continue
        for sp in splits:
            d = df[df.split == sp].copy(); d["window"] = "T"; r = {"learner": name, "split": sp}
            for m in ("auc", "acc", "ll", "far_auc"): s = T.summarize(T.per_seed(d, name, base, "T", m)); r[m] = s["gain"]; r[m + "_p"] = s["p"]
            ref.append(r)
    out += ["\n**Reference learners minus BASE**\n", T.md(pd.DataFrame(ref))]
    far = []
    for c in cands:
        for sp in splits:
            d = df[df.split == sp].copy(); d["window"] = "T"; r = {"learner": c, "split": sp}
            for m in ("far_auc", "far_acc"): s = T.summarize(T.per_seed(d, c, base, "T", m)); r[m] = s["gain"]; r[m + "_ci"] = f"[{s['lo']:+.3f},{s['hi']:+.3f}]"
            far.append(r)
    out += ["\n**Anchor-far stratum**\n", T.md(pd.DataFrame(far))]
    return "\n".join(out)


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text()); blocks = spec["blocks"]; boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 1000
    for b in blocks:
        print(f"## Block {b['name']} (seeds {b['seeds']})\n"); print(forward_tables(spec, b["forward"], boot))
        if b.get("standard"): print(f"\n### {b['name']}: standard random splits (secondary)\n"); print(standard_tables(spec, b["standard"], "standard", boot))
        if b.get("blocked"): print(f"\n### {b['name']}: time-blocked split (secondary)\n"); print(standard_tables(spec, b["blocked"], "blocked", boot))


if __name__ == "__main__": main()

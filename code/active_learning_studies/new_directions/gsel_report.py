"""Generate the pre-registered confirmation tables of PREREGISTRATION_S1.md from the result JSONs (no hand-copied numbers).

    python3 gsel_report.py --out ../pair_disjoint_not_image_disjoint/graph_exploration/TABLES_graph_selection.md

Blocks: conf1 (regular), conf2 (cold), conf3 (HTR).  Families and success rules are those of the pre-registration; dev screens are summarised separately and labelled as such."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import gsel_analyze as ga

HERE = Path(__file__).resolve().parent
R = HERE / "results"
CONTROLS = {"tdc": ["tdc_shuffled", "tdc_typeonly"], "gvopt_lap3@aw": ["gvopt_lap3_shuffled@aw", "vopt_ridge10@aw"], "qcov_alloc": ["qcov_alloc_shuffled", "qcov_alloc_typeonly", "htr_vopt_prio@aw"], "vopt_u@aw": ["vopt_ridge10@aw"]}
GRAPH = {"gvopt_lap3@aw": "gvopt_lap3_shuffled@aw", "tdc": "tdc_shuffled", "qcov_alloc": "qcov_alloc_shuffled"}


def block(name: str, cands, metrics, title: str) -> tuple[str, dict]:
    path = R / name
    if not path.exists() or not any(path.glob("seed*_*.json")): return f"## {title}\n\n(not run yet)\n", {}
    f = ga.load([path]); text = f"## {title}\n\nseeds: {f.seed.nunique()} ({int(f.seed.min())}–{int(f.seed.max())}), arms: {', '.join(sorted(f.selector.unique()))}\n\n"
    cands = [c for c in cands if c in set(f.selector)]; fam = ga.table(f, cands, metrics=metrics)
    text += "### Gain over Random (Holm family = candidates x split x listed metrics)\n\n" + ga.md(fam) + "\n\n"
    comp = ga.table(f, [s for s in sorted(f.selector.unique()) if s not in cands and s != "random"], metrics=metrics)
    text += "### Comparator arms, gain over Random (no Holm; descriptive)\n\n" + ga.md(comp.drop(columns=["holm"]) if "holm" in comp else comp) + "\n\n"
    pairs = [(c, k) for c in cands for k in CONTROLS.get(c, []) if k in set(f.selector)]
    if pairs: text += "### Candidate minus its comparator arm (paired by seed and budget; uncorrected p)\n\n" + ga.md(pd.concat([ga.vs_control(f, c, k, metrics + ("other_acc",)) for c, k in pairs])) + "\n\n"
    other = ga.table(f, cands, metrics=("other_acc", "acc", "ll")); text += "### Other-type accuracy and overall accuracy / log-loss gains (secondary)\n\n" + ga.md(other.drop(columns=["holm"])) + "\n\n"
    text += "### Batch composition (mean over seeds)\n\n" + ga.md(ga.composition(f, ["random"] + [s for s in sorted(f.selector.unique()) if s != "random"])) + "\n\n"
    return text, {"frame": f, "table": fam, "metrics": metrics}


def compact(res: dict, metrics) -> str:
    """One row per candidate: gain [Holm p] for every (split, metric) cell of the pre-registered family."""
    if not res: return "(not run yet)\n\n"
    t = res["table"]; cols = [(sp, m) for m in metrics for sp in ("A", "B")]; rows = []
    for c in t.selector.unique():
        row = {"rule": c}
        for sp, m in cols:
            r = t[(t.selector == c) & (t.split == sp) & (t.metric == m)]; row[f"{m} {sp}"] = f"{r.iloc[0].gain:+.3f} [{r.iloc[0].holm:.4f}]" if len(r) else "n/a"
        rows.append(row)
    return "Gain over Random [Holm-adjusted p]; Holm family = all rows x cells of this block\n\n" + ga.md(pd.DataFrame(rows)) + "\n\n"


def verdicts(res: dict, label: str, graph_metrics) -> str:
    """Apply the pre-registered success rules."""
    out = f"- **{label}**: "
    if not res: return out + "not run yet\n"
    t = res["table"]; lines = []
    for c in sorted(t.selector.unique()):
        g = t[t.selector == c]; ok = {split: all((g[(g.split == split) & (g.metric == m)].holm < .05).any() for m in graph_metrics) for split in ("A", "B")}
        lines.append(f"{c}: Holm<0.05 for {', '.join(graph_metrics)} in A={ok['A']}, B={ok['B']}")
    return out + "; ".join(lines) + "\n"


def pooled(names, cands, metrics, title) -> str:
    """Descriptive pooled estimate (S1 + S2 seeds of one start mode): mean per-seed gain with a seed-bootstrap 95% interval; NOT a test."""
    frames = [ga.load([R / n]) for n in names if (R / n).exists()]; frames = [f for f in frames if not f.empty]
    if len(frames) < 2: return f"### {title}\n\n(S2 not run yet)\n\n"
    f = pd.concat(frames); rng = np.random.default_rng(0); rows = []
    for c in cands:
        g = ga.gains(f, c)
        for split, gg in g.groupby("split"):
            for m in metrics:
                v = gg[m].to_numpy(); boot = [rng.choice(v, len(v)).mean() for _ in range(2000)]; rows.append({"selector": c, "split": split, "metric": m, "n_seeds": len(v), "gain": float(v.mean()), "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
    t = pd.DataFrame(rows); return f"### {title}\n\n" + ga.md(t) + "\n\n"


def mechanism(names, title) -> str:
    """Batch composition averaged over budgets 10-60, seeds and splits: type shares, share of decisive judgments, unique images, plus the HTR/other-type gain over Random for context."""
    frames = [ga.load([R / n]) for n in names if (R / n).exists()]; frames = [f for f in frames if not f.empty]
    if not frames: return ""
    f = pd.concat(frames); c = ga.composition(f, sorted(f.selector.unique())); keep = [x for x in c.columns if x not in ("budget",)]
    agg = c[keep].groupby("selector").mean(numeric_only=True).reset_index(); order = ["random"] + sorted(s for s in agg.selector if s != "random")
    agg = agg.set_index("selector").loc[order].reset_index(); return f"### {title}\n\n" + ga.md(agg) + "\n\n"


def dev_summary() -> str:
    text = "## Development screens (seeds 8000–8019; NOT evidence)\n\nMean gain over Random (mean over budgets, 20 dev seeds), uncorrected Wilcoxon p in the generating JSONs; regenerate with `gsel_analyze.py`.\n\n"
    for d in sorted(p.name for p in R.glob("gs_dev*")):
        f = ga.load([R / d])
        if f.empty: continue
        sels = [s for s in sorted(f.selector.unique()) if s != "random"]; t = ga.table(f, sels, metrics=("acc", "auc", "htr_acc", "htr_auc"))
        piv = t.pivot_table(index="selector", columns=["split", "metric"], values="gain"); piv.columns = [f"{s}:{m}" for s, m in piv.columns]; piv = piv.reset_index()
        text += f"### {d}\n\n" + ga.md(piv) + "\n\n"
    return text


def oracle_section() -> str:
    """Diagnostic ceiling (dev seeds; reads hidden outcomes): decisive-enrichment with the true decisive flag."""
    out = ""
    for d, title in (("gs_dev6_oracle", "regular start"), ("gs_dev6_oracle_cold", "cold start")):
        f = ga.load([R / d]) if (R / d).exists() else pd.DataFrame()
        if f.empty: continue
        sels = [x for x in sorted(f.selector.unique()) if x != "random"]; t = ga.table(f, sels, metrics=("acc", "auc", "htr_acc", "htr_auc"))
        piv = t.pivot_table(index="selector", columns=["split", "metric"], values="gain"); piv.columns = [f"{s_}:{m}" for s_, m in piv.columns]; piv = piv.reset_index()
        c = ga.composition(f, sels).groupby("selector")[["decisive_share"]].mean().reset_index(); out += f"### ORACLE diagnostic, {title} (dev seeds 8000–8019; reads hidden outcomes, not a method)\n\n" + ga.md(piv) + "\n\nDecisive share of the batches:\n\n" + ga.md(c) + "\n\n"
    return out or "(oracle diagnostic not run yet)\n\n"


def sections() -> dict:
    amet = ("acc", "auc"); cands = ["tdc", "gvopt_lap3@aw", "vopt_u@aw", "qcov_alloc"]; htr = ("htr_acc", "htr_auc"); sec = {}
    t1, r1 = block("gs_conf1", cands, amet, "Confirmation 1: regular start, seeds 8100–8134, family A (16 tests)")
    t2, r2 = block("gs_conf2_cold", cands, amet, "Confirmation 2: cold start (10 random initial judgments), seeds 8200–8234, family A (16 tests)")
    t3, r3 = block("gs_conf3_htr", cands, htr, "Confirmation 3: HTR block, seeds 8300–8334, family B (16 tests)")
    s2c, rc = block("gs_s2_cold", ["tdc", "qcov_alloc"], amet, "Replication S2 (cold start), seeds 8340–8369, family of 8 tests")
    s2r, rr = block("gs_s2_regular", ["tdc", "qcov_alloc"], amet, "Replication S2 (regular start), seeds 8370–8399, family of 8 tests")
    sec["verdicts"] = verdicts(r1, "conf1 regular, acc+auc", amet) + verdicts(r2, "conf2 cold, acc+auc", amet) + verdicts(r3, "conf3 HTR, htr_acc+htr_auc", htr) + verdicts(rc, "S2 cold, acc+auc", amet) + verdicts(rr, "S2 regular, acc+auc", amet) + "\n"
    for k, res, mets in (("compact_conf1", r1, amet), ("compact_conf2", r2, amet), ("compact_conf3", r3, htr), ("compact_s2c", rc, amet), ("compact_s2r", rr, amet)): sec[k] = compact(res, mets)
    sec["full"] = t1 + t2 + t3 + s2c + s2r
    sec["mech1"] = mechanism(["gs_conf1"], "Confirmation 1 (regular start)"); sec["mech2"] = mechanism(["gs_conf2_cold"], "Confirmation 2 (cold start)"); sec["mech3"] = mechanism(["gs_conf3_htr"], "Confirmation 3 (HTR block)")
    sec["pooled"] = pooled(["gs_conf1", "gs_s2_regular"], ["tdc", "qcov_alloc"], amet, "Regular start (seeds 8100–8134 and 8370–8399)") + pooled(["gs_conf2_cold", "gs_s2_cold"], ["tdc", "qcov_alloc"], amet, "Cold start (seeds 8200–8234 and 8340–8369)")
    sec["oracle"] = oracle_section(); sec["dev"] = dev_summary(); return sec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--template", default=None); ap.add_argument("--doc", default=None); a = ap.parse_args(); sec = sections()
    if a.out:
        text = "# Tables for the graph-selection study (generated by gsel_report.py; do not edit by hand)\n\n## Pre-registered verdict inputs (Holm < 0.05 in all four cells = both splits, both metrics)\n\n" + sec["verdicts"]
        text += "## Compact view of the pre-registered cells\n\n" + "".join(f"### {t}\n\n" + sec[k] for t, k in (("Confirmation 1 (regular start, 8100–8134)", "compact_conf1"), ("Confirmation 2 (cold start, 8200–8234)", "compact_conf2"), ("Confirmation 3 (HTR, 8300–8334)", "compact_conf3"), ("Replication S2 cold (8340–8369)", "compact_s2c"), ("Replication S2 regular (8370–8399)", "compact_s2r")))
        text += sec["full"] + "## Mechanism: what the batches contain (mean over budgets, seeds and splits)\n\n" + sec["mech1"] + sec["mech2"] + sec["mech3"] + "## Pooled descriptive estimates (S1 + S2 seeds; not tests)\n\n" + sec["pooled"] + "## Oracle diagnostic\n\n" + sec["oracle"] + sec["dev"]
        Path(a.out).write_text(text); print(f"wrote {a.out} ({len(text)} chars)")
    if a.template and a.doc:
        t = Path(a.template).read_text()
        for k, v in sec.items(): t = t.replace(f"<!--TABLE:{k}-->", v)
        Path(a.doc).write_text(t); print(f"wrote {a.doc}")


if __name__ == "__main__":
    main()

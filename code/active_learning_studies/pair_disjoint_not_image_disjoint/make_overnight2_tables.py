#!/usr/bin/env python3
"""Markdown tables for report section 5.20 from the overnight-2 cells: usage make_overnight2_tables.py P1|P2|P3  (prints markdown)."""
import sys

import numpy as np

import aggregate_overnight2 as a

LABEL = {"typed_decisive_coverage": "Type-aware graph coverage x P(decisive)", "typed_decisive_coverage_unc": "... x own-head uncertainty", "typed_decisive_bald": "Laplace BALD x P(decisive), graph features",
         "typed_decisive_coverage_typeonly": "(control) decisive predictor from the type only", "typed_decisive_coverage_shuffled": "(control) type posterior shuffled over images",
         "vopt_u": "(reference) pool-wide I-optimal design, no graph (vopt_u)", "core_set_relation": "(reference) Core-set, relation-aware pairs",
         "gvopt_lap3": "Laplacian-regularised I-optimal design (lam 3)", "gvopt_lap3_shuffled": "(control) same, graph shuffled", "gvopt_lap": "Laplacian-regularised design (lam 1)",
         "gvopt_lap10": "Laplacian-regularised design (lam 10)", "gvopt_lap03": "Laplacian-regularised design (lam 0.3)", "gvopt_type": "I-optimal design, type-posterior pool weights",
         "gvopt_prop": "I-optimal design on propagated features", "gvopt_sigma": "Sigma-optimal design", "gvopt_lapsigma": "Laplacian-regularised Sigma-optimal design"}
CAND = {"P1": ["typed_decisive_coverage", "typed_decisive_coverage_unc", "typed_decisive_bald"], "P3": ["gvopt_lap3", "typed_decisive_coverage"]}
ORDER = {"P1": ["typed_decisive_coverage", "typed_decisive_coverage_unc", "typed_decisive_bald", "typed_decisive_coverage_typeonly", "typed_decisive_coverage_shuffled", "vopt_u", "core_set_relation"],
         "P3": ["gvopt_lap3", "typed_decisive_coverage", "gvopt_lap3_shuffled", "vopt_u"]}
PAIRS = {"P1": [("typed_decisive_coverage", "typed_decisive_coverage_typeonly"), ("typed_decisive_coverage", "typed_decisive_coverage_shuffled"), ("typed_decisive_coverage", "vopt_u")],
         "P3": [("gvopt_lap3", "gvopt_lap3_shuffled"), ("gvopt_lap3", "vopt_u"), ("gvopt_lap3", "typed_decisive_coverage")]}


def table(phase):
    out = a.DIRS[phase]; cand = CAND[phase]; rows = ["| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    diffs = ["| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for split, cond in a.BLOCKS:
        res = {m: a.gains(out, split, cond, m) for m in a.METRICS}; ll = res["test_decisive_log_loss"]
        if ll is None: continue
        tests = {c: a.wp(ll[c].dropna().to_numpy()) for c in ll.columns}; hp = a.holm({c: tests[c] for c in cand if c in tests and not np.isnan(tests[c])}); cn = "single-shot" if cond == "single" else "sequential"
        for c in ORDER[phase]:
            if c not in ll: continue
            v = ll[c].dropna().to_numpy()
            rows.append(f"| {split} | {cn} | {LABEL[c]} | {v.mean():+.3f} | {a.fp(tests[c])} | {a.fp(hp[c]) if c in hp else ''} | {res['test_decisive_auc'][c].mean():+.4f} | {res['test_decisive_accuracy'][c].mean():+.4f} | {(v > 0).mean():.0%} | {len(v)} |")
        for x, y in PAIRS[phase]:
            if x in ll and y in ll:
                cells = []
                for m in a.METRICS:
                    d = (res[m][x] - res[m][y]).dropna().to_numpy(); cells += [f"{d.mean():+.4f}", a.fp(a.wp(d))]
                diffs.append(f"| {split} | {cn} | {LABEL[x]} minus {LABEL[y]} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n\n" + "\n".join(diffs)


if __name__ == "__main__":
    print(table(sys.argv[1]))

"""Apply the candidate-selection rule of PREREGISTRATION_HF.md to the development results.

  python3 hf_select.py   ->  results/hf/dev_selection.md

Conditions: {Split A, Split B} x {standard endpoint, anchor-strict endpoint (delta 0.01)}.  A variant is eligible if in all four conditions the HTR accuracy gain >= 0, every other type's accuracy change >= -0.01
and the all-type AUC change >= -0.002.  Candidate = eligible variant with the largest minimum HTR log-loss gain over the four conditions; variants within 0.005 of the best are resolved by the fewest fitted components
(blend < coupling < features < teacher).
"""
from pathlib import Path

import pandas as pd

import hf_gain as g
import nd_core as core

HF = core.HERE / "results" / "hf"; BASE = "lap3_k10_st4"
VARIANTS = ["hfb_g5", "hfb_g10", "hfb_g20", "hfb3_g10", "hfs", "hfq1", "hfc1", "hfcb10"]; COMPLEXITY = {"hfb": 0, "hfc": 1, "hfs": 2, "hfq": 3}


def conditions() -> pd.DataFrame:
    rows = []
    for tag, groups in (("standard", [("a1", "s2_blend"), ("a1", "s2_learned")]), ("strict0.01", [("s2s_blend",), ("s2s_learned",)])):
        for dirs in groups:
            d = g.read([HF / x for x in dirs]); cands = [c for c in VARIANTS if c in set(d.learner)]
            if cands: r = g.compare(d, BASE, cands); r["endpoint"] = tag; rows.append(r)
    return pd.concat(rows, ignore_index=True)


def main():
    r = conditions(); out = []
    for c in VARIANTS:
        x = r[r.cand == c]
        if len(x) != 4: out.append({"variant": c, "eligible": False, "reason": f"{len(x)}/4 conditions"}); continue
        eligible = bool((x.acc_HTR >= 0).all() and all((x[f"acc_{t}"] >= -0.01).all() for t in ("1x1", "c6x2", "RT13")) and (x.auc_all >= -0.002).all())
        out.append({"variant": c, "eligible": eligible, "min_ll_gain": float(x.ll_HTR.min()), **{f"ll_{e}_{s}": float(x[(x.endpoint == e) & (x.split == s)].ll_HTR.iloc[0]) for e in ("standard", "strict0.01") for s in ("A", "B")},
                    **{f"acc_{e}_{s}": float(x[(x.endpoint == e) & (x.split == s)].acc_HTR.iloc[0]) for e in ("standard", "strict0.01") for s in ("A", "B")}, "min_other_acc": float(min(x[f"acc_{t}"].min() for t in ("1x1", "c6x2", "RT13"))), "min_auc": float(x.auc_all.min())})
    t = pd.DataFrame(out); text = "# Development selection (seeds 10400-10419)\n\n" + t.round(4).to_markdown(index=False) + "\n\n"
    el = t[t.eligible.astype(bool)] if "eligible" in t else t.iloc[0:0]
    if len(el):
        best = el.min_ll_gain.max(); near = el[el.min_ll_gain >= best - 0.005].copy(); near["complexity"] = [COMPLEXITY[v[:3]] for v in near.variant]; pick = near.sort_values(["complexity", "min_ll_gain"], ascending=[True, False]).iloc[0].variant
        text += f"Eligible: {list(el.variant)}.  Best minimum log-loss gain {best:.4f}; within 0.005: {list(near.variant)}.  **Candidate: `{pick}`.**\n"
    else: text += "**No variant is eligible; no candidate is confirmed.**\n"
    (HF / "dev_selection.md").write_text(text); r.to_csv(HF / "dev_selection_conditions.csv", index=False); print(text)


if __name__ == "__main__": main()

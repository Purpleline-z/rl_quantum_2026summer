"""HTR failure analysis, label side (no learner): how consistent are the labels themselves?

  python3 hf_labels.py   ->  results/hf/labels_report.md  (+ results/hf/label_neighbours.csv)

Everything here uses only the raw judgments and the frozen SimCLR geometry of the 284 judged images.  Facts it establishes:
 * how many distinct decisive judgments of each type exist, whether any (pair, type) was judged twice (it was not), whether labellers overlap (no pair was judged by two labellers);
 * label agreement of a judgment with its nearest *other* judgment of the same type as a function of the (oriented) distance between the two comparisons: if near-identical comparisons were labelled
   differently, no smooth model can fit both (a noise ceiling);
 * 3-cycles / merged-image conflicts of the directed "wins" graph (transitivity violations);
 * accuracy of model-free rules (time order, temperature order, similarity to the ideal HTR images) on the HTR judgments, as a baseline for the learner.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import nd_core as core

OUT = core.HERE / "results" / "hf"


def load():
    j = pd.read_csv(OUT / "judgments.csv"); img = pd.read_csv(OUT / "images.csv"); dm = np.load(OUT / "img_dist.npy"); return j, img, dm


def oriented_neighbours(d: pd.DataFrame, dm: np.ndarray) -> pd.DataFrame:
    """For every judgment of ``d`` (decisive, one type) its nearest other judgment: D = max over the two images of the distance to the matching image, best of the two orientations."""
    a, b, s = d.i1.to_numpy(), d.i2.to_numpy(), d.sign.to_numpy(); n = len(d); out = []
    for x in range(n):
        best = (9.0, -1, 0.0)
        for y in range(n):
            if y == x: continue
            same = max(dm[a[x], a[y]], dm[b[x], b[y]]); flip = max(dm[a[x], b[y]], dm[b[x], a[y]])
            D, agree = (same, s[x] * s[y]) if same <= flip else (flip, -s[x] * s[y])
            if D < best[0]: best = (D, y, agree)
        out.append({"jid": int(d.jid.iloc[x]), "nn_jid": int(d.jid.iloc[best[1]]), "nn_D": best[0], "nn_agree": best[2]})
    return pd.DataFrame(out)


def all_pair_agreement(d: pd.DataFrame, dm: np.ndarray) -> pd.DataFrame:
    a, b, s = d.i1.to_numpy(), d.i2.to_numpy(), d.sign.to_numpy(); rows = []
    for x, y in itertools.combinations(range(len(d)), 2):
        same = max(dm[a[x], a[y]], dm[b[x], b[y]]); flip = max(dm[a[x], b[y]], dm[b[x], a[y]])
        D, agree = (same, s[x] * s[y]) if same <= flip else (flip, -s[x] * s[y]); rows.append((D, agree))
    return pd.DataFrame(rows, columns=["D", "agree"])


def cycles(d: pd.DataFrame, merge: np.ndarray | None = None) -> dict:
    """Directed wins graph (winner -> loser) on images (or on merged image clusters); counts 3-cycles and 2-cycles (the same cluster pair judged both ways)."""
    w = [(int(r.i1), int(r.i2)) if r.sign > 0 else (int(r.i2), int(r.i1)) for r in d.itertuples()]
    if merge is not None: w = [(int(merge[a]), int(merge[b])) for a, b in w]
    w = [(a, b) for a, b in w if a != b]; edges = set(w); two = sum(1 for a, b in edges if (b, a) in edges) // 2; nodes = sorted({x for e in edges for x in e}); adj = {x: set() for x in nodes}
    for a, b in edges: adj[a].add(b)
    three = sum(1 for a in nodes for b in adj[a] for c in adj[b] if a in adj[c] and a < b and a < c)
    return {"edges": len(edges), "nodes": len(nodes), "2-cycles": two, "3-cycles": three}


def clusters(dm: np.ndarray, delta: float) -> np.ndarray:
    n = len(dm); lab = np.arange(n)
    def find(x):
        while lab[x] != x: lab[x] = lab[lab[x]]; x = lab[x]
        return x
    for x, y in zip(*np.nonzero(np.triu(dm < delta, 1))): lab[find(x)] = find(y)
    return np.array([find(x) for x in range(n)])


def rule_accuracy(h: pd.DataFrame) -> dict:
    """Model-free rules on the decisive HTR judgments (sign = +1 if image 1 won)."""
    s = h.sign.to_numpy(); out = {}
    out["image 2 listed second wins (always -1)"] = float((s < 0).mean()); out["image 1 wins (always +1)"] = float((s > 0).mean())
    ok = h.same_run.to_numpy(); out["later frame wins (same-run pairs only, n=%d)" % ok.sum()] = float((np.sign(h.idx1 - h.idx2)[ok] == s[ok]).mean()) if ok.any() else float("nan")
    out["higher temperature wins (all)"] = float((np.sign(h.temp1 - h.temp2) == s).mean()); out["lower temperature wins (all)"] = float((np.sign(h.temp2 - h.temp1) == s).mean())
    out["closer to ideal HTR wins"] = float((np.sign(h.d_HTR2 - h.d_HTR1) == s).mean())
    out["farther from ideal RT13 wins"] = float((np.sign(h.d_RT131 - h.d_RT132) == s).mean())
    out["(d_RT13 - d_HTR) larger wins"] = float((np.sign((h.d_RT131 - h.d_HTR1) - (h.d_RT132 - h.d_HTR2)) == s).mean())
    return out


def main():
    j, img, dm = load(); lines = ["# HTR label-consistency report (static: labels and frozen SimCLR geometry only)\n"]
    lines.append("## 1. What data there is\n"); t = j.groupby("type").winner.value_counts().unstack(1).fillna(0).astype(int); lines.append(t.to_markdown() + "\n")
    dup = j.groupby(["pair", "type"]).size(); pairs = j.assign(u=[tuple(sorted(x)) for x in zip(j.img1, j.img2)]); dup2 = pairs.groupby(["u", "type"]).size()
    lines.append(f"* (pair, type) judged more than once: **{int((dup > 1).sum())}** (unordered image pairs judged more than once per type: {int((dup2 > 1).sum())}) -> no repeated comparisons, no inter-labeller overlap exist; the ceiling cannot come from repeats.")
    h = j[(j.type == "HTR") & j.decisive].reset_index(drop=True)
    lines.append(f"* decisive HTR judgments: **{len(h)}** (image 1 won {int((h.sign > 0).sum())}, image 2 won {int((h.sign < 0).sum())}); not_apply {int(((j.type == 'HTR') & (j.winner == 'not_apply')).sum())}, tie {int(((j.type == 'HTR') & (j.winner == 'tie')).sum())}; distinct images in decisive HTR judgments: {len(set(h.i1) | set(h.i2))}.")
    lines.append(f"* labellers (decisive HTR): {h.labeller.value_counts().to_dict()}; confidence {h.confidence.value_counts().to_dict()}; runs of the two images: {pd.Series(list(h.run1) + list(h.run2)).value_counts().to_dict()}; both frames from the same run: {int(h.same_run.sum())} of {len(h)}.")
    lines.append("\nComparison kind by nearest ideal-image class of the two frames (decisive judgments of each type; 1-NN on centred SimCLR features, the four active classes):\n")
    lines.append(j[j.decisive].groupby("type").cls_pair.value_counts().unstack(0).fillna(0).astype(int).to_markdown() + "\n")

    lines.append("## 2. Do near-identical comparisons agree? (oriented nearest-neighbour label agreement, same type, decisive)\n")
    lines.append("D = max over the two matched images of 1 - cos (SimCLR); the typical distance between a test frame and its nearest training frame is 0.0002-0.02, between different-type ideal images 0.58 (median).\n")
    rows = []; allnb = []
    for typ, d in j[j.decisive].groupby("type"):
        d = d.reset_index(drop=True); nb = oriented_neighbours(d, dm); nb["type"] = typ; allnb.append(nb); ap = all_pair_agreement(d, dm)
        for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 9)):
            m = ap[(ap.D >= lo) & (ap.D < hi)]; rows.append({"type": typ, "D bin": f"{lo}-{hi}", "n judgment pairs": len(m), "label agreement": float((m.agree > 0).mean()) if len(m) else float("nan")})
    lines.append(pd.DataFrame(rows).pivot(index="D bin", columns="type", values=["n judgment pairs", "label agreement"]).round(3).to_markdown() + "\n")
    nb = pd.concat(allnb); nb.to_csv(OUT / "label_neighbours.csv", index=False)
    lines.append("Leave-one-out 1-NN **label predictor** (predict a judgment's winner from its nearest other same-type comparison, oriented; uses no model, no image-disjointness):\n")
    rows = []
    for typ, g in nb.groupby("type"):
        rows.append({"type": typ, "n": len(g), "1-NN accuracy": float((g.nn_agree > 0).mean()), "median nn_D": float(g.nn_D.median()), "accuracy if nn_D<0.1": float((g[g.nn_D < 0.1].nn_agree > 0).mean()), "n(nn_D<0.1)": int((g.nn_D < 0.1).sum())})
    lines.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

    lines.append("## 3. Transitivity (directed winner -> loser graph, HTR decisive judgments)\n")
    rows = []
    for delta in (0.0, 0.003, 0.01, 0.03, 0.1):
        merge = None if delta == 0 else clusters(dm, delta); c = cycles(h, merge); c["merge radius"] = delta; c["n clusters used"] = c["nodes"]; rows.append(c)
    lines.append(pd.DataFrame(rows)[["merge radius", "nodes", "edges", "2-cycles", "3-cycles"]].to_markdown(index=False) + "\n")
    lines.append("(2-cycle = the same pair of merged image clusters judged in both directions; 3-cycle = A>B, B>C, C>A.)  The graph is almost a matching, so transitivity has little power.\n")

    lines.append("## 4. Model-free rules on the decisive HTR judgments (accuracy; sign +1 = image 1 won)\n")
    lines.append(pd.Series(rule_accuracy(h)).round(3).to_frame("accuracy").to_markdown() + "\n")
    (OUT / "labels_report.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__": main()

"""Unfitted answerability heuristics: AUC of each single propagated-posterior feature for decisive / tie / not_apply over all 521 rows (no model is fitted, so no CV needed;
the choice among 8 features x graph settings is design-informing only). Reuses diagnostic_outcomes.py."""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
import diagnostic_outcomes as D

names = ["q_a(t)", "q_b(t)", "|qa-qb|(t)", "qa+qb(t)", "min", "max", "L1(qa,qb)", "qa.qb"]
ys = {o: (D.d.Winner.isin(["1", "2"]) if o == "decisive" else D.d.Winner == o).values.astype(int) for o in ("decisive", "tie", "not_apply")}
B, U = D.pair_nodes + D.ideal, D.pair_nodes + D.ideal + D.unl
for label, nodes, k, alpha in [("base k10 a.9", B, 10, .9), ("base k10 a.99", B, 10, .99), ("base k20 a.999", B, 20, .999), ("+unl k40 a.99", U, 40, .99), ("+unl k40 a.999", U, 40, .999)]:
    out = []
    for r in range(5):
        rng = np.random.default_rng(r); refs = {}
        for c in D.LAB.values():
            ids = [i for i in D.ideal if D.LAB[D.man.label[i]] == c]; ids = list(rng.permutation(ids)); refs.update({int(i): c for i in ids[:int(round(.6 * len(ids)))]})
        q, pos = D.graph(nodes, k, alpha, refs); F = D.feats(q, pos); out.append([[roc_auc_score(y, F[:, j]) for j in range(8)] for y in ys.values()])
    m = np.mean(out, 0); print(f"\n{label}: single-feature AUC (rows: decisive, tie, not_apply; columns: {names})")
    for o, row in zip(ys, m): print(f"  {o:10s}" + "".join(f"{v:7.3f}" for v in row))

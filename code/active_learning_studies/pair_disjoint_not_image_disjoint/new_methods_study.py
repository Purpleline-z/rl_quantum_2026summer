#!/usr/bin/env python3
"""Runner for the new methods: judgment_unit_study with extra row rules registered (no change to the existing study code)."""
import judgment_unit_study as study
import judgment_unit_strategies as ju
import new_methods_strategies as nm

ju.ROW_RULES.update(nm.NEW); ju.STRATEGY_FAMILY = ju.STRATEGY_FAMILY + tuple(n for n in nm.NEW if n not in ju.STRATEGY_FAMILY)
_original_init = study.Context.__init__
def _init(self, *a, **k):
    _original_init(self, *a, **k); nm.CTX["ctx"] = self
study.Context.__init__ = _init

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
_original_evaluate = study.Context.evaluate
TYPE_NAMES = {0: "t1x1", 2: "tc6x2", 3: "t13", 4: "htr"}
def _evaluate(self, model):
    out = _original_evaluate(self, model)
    rows = self.exp.rows_for(self.test); rows = rows[rows.Winner.isin(["1", "2"])]; head = model.reward_head.eval()
    with torch.no_grad(): a, b = head(self.features.get(rows.resolved_img1)), head(self.features.get(rows.resolved_img2))
    d = (a - b)[torch.arange(len(rows)), torch.as_tensor(rows.type_idx.to_numpy())].numpy(); s = np.where((rows.Winner == "1").to_numpy(), 1.0, -1.0); typ = rows.type_idx.to_numpy()
    for k, name in TYPE_NAMES.items():
        m = typ == k; out[f"{name}_rows"] = int(m.sum())
        out[f"{name}_acc"] = float(((d[m] * s[m]) > 0).mean()) if m.any() else float("nan")
        out[f"{name}_ll"] = float(np.logaddexp(0.0, -s[m] * d[m]).mean()) if m.any() else float("nan")
        out[f"{name}_auc"] = float(roc_auc_score(s[m] > 0, d[m])) if m.any() and len(set(s[m])) == 2 else float("nan")
    return out
study.Context.evaluate = _evaluate


if __name__ == "__main__":
    study.main()

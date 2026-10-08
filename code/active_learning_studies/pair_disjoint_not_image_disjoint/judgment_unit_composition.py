#!/usr/bin/env python3
"""What the strategies reveal: mean number of distinct pairs touched and share of decisive judgments among the revealed ones (budget 60 by default).
usage: judgment_unit_composition.py <run> <condition> [budget]"""
import json, sys
import numpy as np
import pandas as pd
import judgment_unit_study as study
run, condition = sys.argv[1], sys.argv[2]; budget = sys.argv[3] if len(sys.argv) > 3 else "60"; base = study.OUT / run
manifests = {int(p.stem.split("seed")[1]): json.loads(p.read_text()) for p in base.glob("manifest_seed*.json")}
rows = []
for path in (base / condition).glob("seed*_*.json"):
    d = json.loads(path.read_text())
    if d["strategy"] == "initial_only": continue
    sel = d["checkpoints"][budget]["selected"]; pool = manifests[d["seed"]]["pool"]
    rows.append({"family": study.family_of(d["strategy"]), "pairs": d["checkpoints"][budget]["n_pairs_touched"], "decisive": np.mean([pool[i][3] in ("1", "2") for i in sel]), "tie": np.mean([pool[i][3] == "tie" for i in sel])})
out = pd.DataFrame(rows).groupby("family").mean().sort_values("pairs"); out.to_csv(base / "analysis" / f"composition_{condition}_budget{budget}.csv")
print(out.round(3).to_string())

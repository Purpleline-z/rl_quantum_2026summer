#!/usr/bin/env python3
"""What do the selected judgments look like?  usage: new_methods_mechanism.py <run> <comma list of strategies>  (budget 60, single-shot cells)
Decisive / tie / not_apply shares, share per type, distinct pairs touched; Random for comparison (5 draws pooled)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent / "results" / "new_methods"; run = sys.argv[1]; names = sys.argv[2].split(",")
rows = []
for split in "AB":
    for p in sorted((ROOT / run / split / "single").glob("seed*_*.json")):
        d = json.loads(p.read_text()); s = d["strategy"]; fam = s.rsplit("_r", 1)[0] if s.startswith("random") and s[-1].isdigit() and "_r" in s else s
        if fam not in names + ["random"] or s == "initial_only": continue
        man = json.loads((ROOT / run / split / f"manifest_seed{d['seed']}.json").read_text()); pool = man["pool"]
        for b in ("20", "60"):
            sel = d["checkpoints"][b]["selected"]; w = [pool[i][3] for i in sel]; t = [pool[i][2] for i in sel]
            rows.append({"split": split, "strategy": fam, "budget": int(b), "decisive": np.mean([x in ("1", "2") for x in w]), "tie": np.mean([x == "tie" for x in w]), "not_apply": np.mean([x == "not_apply" for x in w]),
                         "pairs": d["checkpoints"][b]["n_pairs_touched"], **{f"type{k}": np.mean([x == k for x in t]) for k in (0, 2, 3, 4)}})
t = pd.DataFrame(rows).groupby(["split", "budget", "strategy"]).mean().round(3); print(t.to_string())

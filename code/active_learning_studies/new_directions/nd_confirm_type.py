import json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from nd_confirm import holm
d = pd.DataFrame(json.load(open(sys.argv[1]))); base = d[d.learner == "baseline"].groupby(["seed", "budget"]).type_acc.mean(); rows = []
for l in ("aw2", "aw8", "aw32"):
    diff = (d[d.learner == l].groupby(["seed", "budget"]).type_acc.mean() - base).groupby("seed").mean(); rng = np.random.default_rng(0); b = rng.choice(diff.values, (4000, len(diff))).mean(1)
    rows.append({"learner": l, "n_seeds": len(diff), "gain": diff.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5), "share_better": float((diff > 0).mean()), "p": float(wilcoxon(diff).pvalue)})
out = pd.DataFrame(rows); out["holm"] = holm(out.p); pd.set_option("display.width", 200); print(out.round(4).to_string(index=False))
print(d.groupby(["budget", "learner"]).type_acc.mean().unstack().round(3))

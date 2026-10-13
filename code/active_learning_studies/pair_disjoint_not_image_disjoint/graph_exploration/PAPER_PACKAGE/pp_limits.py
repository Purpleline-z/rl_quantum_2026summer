"""Numbers for the 'assumptions and limits' section, all computed from stored files:
  data facts      from results/pair_graph_summary.csv: labelled pair groups, distinct images, connected components of the comparison graph, judgments by outcome, images per session date
  test sizes      mean number of held-out decisive judgments per (seed, split) (sh/c1_learn) and per type
  re-splitting    number of seeds vs the fixed 168 groups; number of confirmation seeds (35-seed blocks) used by the package
  verdict counts  how many pre-registered success rules were met / not met (tables/verdicts.csv)
  winner's curse  median / min / max of confirmation-to-development gain ratios (tables/winners_curse.csv)
Outputs tables/limits_facts.csv, numbers_limits.json."""
import re

import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import Numbers
from pp_registry import SEED_BLOCKS


def main():
    N = Numbers("limits"); rows = []
    pg = pd.read_csv(L.STUDY / "results" / "pair_graph_summary.csv"); imgs = {}
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for pid in pg.pair_id:
        a, b = [s.strip() for s in pid.split("||")]; imgs.setdefault(a, 0); imgs.setdefault(b, 0); parent[find(a)] = find(b)
    comps = len({find(x) for x in imgs}); dates = pd.Series([re.search(r"Trajectories/(\d{4}-\d{2}-\d{2})", p).group(1) for p in imgs]).value_counts()
    outcomes = pd.Series([w for ws in pg.winners for w in str(ws).split(",")]).value_counts()
    facts = {"data.pair_groups": len(pg), "data.images": len(imgs), "data.components": comps, "data.judgments": int(pg.rows.sum()), "data.sessions": int(len(dates)), "data.decisive_share": float(outcomes[["1", "2"]].sum() / outcomes.sum()),
             "data.tie_share": float(outcomes.get("tie", 0) / outcomes.sum()), "data.not_apply_share": float(outcomes.get("not_apply", 0) / outcomes.sum())}
    for i, (d, c) in enumerate(dates.items()): facts[f"data.session{i + 1}_images"] = int(c)
    f = L.load_dir("sh/c1_learn"); f = f[f.learner == "aw8_all"]
    for s in ("A", "B"): facts[f"test.n_decisive.{s}"] = float(f[(f.split == s)].groupby(["seed", "budget", "draw"]).n_test.first().mean())
    facts["seeds.n_blocks"] = len(SEED_BLOCKS); facts["seeds.confirmation_blocks_35"] = sum(1 for _, role, a, b in SEED_BLOCKS if b - a + 1 == 35); facts["seeds.confirmation_seeds_total"] = sum(b - a + 1 for _, role, a, b in SEED_BLOCKS if "confirmation" in role)
    facts["seeds.development_seeds_total"] = sum(b - a + 1 for _, role, a, b in SEED_BLOCKS if "development" in role)
    v = pd.read_csv(L.TABLES / "verdicts.csv"); ok = v.verdict.isin(["CONFIRMED", "MET", "ALL SIGNIFICANT"]); facts["verdicts.total"] = len(v); facts["verdicts.met"] = int(ok.sum()); facts["verdicts.not_met"] = int((~ok).sum())
    wc = pd.read_csv(L.TABLES / "winners_curse.csv"); ratios = np.concatenate([wc[c].dropna().to_numpy() for c in wc.columns if c.endswith("_ratio")])
    facts["wc.ratio_median"] = float(np.median(ratios)); facts["wc.ratio_min"] = float(ratios.min()); facts["wc.ratio_max"] = float(ratios.max()); facts["wc.n_cells"] = len(ratios); facts["wc.share_below_1"] = float((ratios < 1).mean())
    a6 = (L.RES / "rt" / "a6.md").read_text()   # red-team identity audit: distance scale between reference images (1 - cos), stored table
    for kind, key in (("same type", "same_type"), ("different type", "diff_type")):
        m = re.search(rf"\| {kind} \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", a6); facts[f"ref.{key}.n_pairs"] = int(m.group(1)); facts[f"ref.{key}.median"] = float(m.group(2)); facts[f"ref.{key}.q05"] = float(m.group(3))
    for k, val in facts.items(): N.put(k, val); rows.append(dict(key=k, value=val))
    pd.DataFrame(rows).to_csv(L.TABLES / "limits_facts.csv", index=False); N.save(); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in facts.items()})


if __name__ == "__main__":
    main()

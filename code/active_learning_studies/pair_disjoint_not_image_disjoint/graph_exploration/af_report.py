"""Generate the tables of RESULTS_anchor_far.md (every number comes from the result files; narrative lives in RESULTS_anchor_far.md and refers to these tables).

  python3 af_report.py > tables_anchor_far.md
Sections: A headroom (red-team seeds 8800-8819), B dev screen rounds 1-2 and blocked, C confirmation 1 tests / verdicts / dose-response / strata / cluster-aware, D variant log.
Missing result directories are skipped (the section prints 'not available').
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

import af_tables as T
import af_confirm as C

HERE = Path(__file__).resolve().parent
RT = HERE.parents[1] / "new_directions" / "results" / "rt"
SPEC = HERE / "af_spec_AF1.json"


def have(*names): return all((RT / n).exists() and any((RT / n).iterdir()) for n in names)


def section(title, body): print(f"\n## {title}\n\n{body}\n")


def main():
    import json
    spec = json.loads(SPEC.read_text()); ref = spec["reference"]
    # A. headroom on the red-team data
    if have("a1", "a1b"):
        df = T.load(str(RT / "a1"), str(RT / "a1b"))
        section("A1. Absolute AUC / accuracy by distance to the nearest typed anchor — red-team seeds 8800–8819 (random label sets, budgets 10/20/40/60, 3 draws)", T.to_md(T.absolute_table(df, ["lap3_k10_st4", "aw8_all_st2lr0.5"])))
    # B. development screen
    if have("af_d1", "af_d1b"):
        d12 = ",".join(str(RT / n) for n in ("af_d1", "af_d2") if (RT / n).exists())
        df = T.load(d12, str(RT / "af_d1b"))
        section("B1. Development screen (seeds 10800–10811, budgets 20/40, 2 draws): gain over af_base (= lap3_k10_st4); far = both images > 0.01 from every typed anchor; two-sided Wilcoxon p over seeds, NOT adjusted", T.to_md(T.dev_table(df, ref)))
        names = sorted(df.learner.unique()); section("B2. Learners screened (variant log)", ", ".join(f"`{n}`" for n in names))
    if have("af_b1"):
        j = C.load_json(str(RT / "af_b1")); rows = []
        for l in sorted(j.learner.unique()):
            if l == ref: continue
            r = {"learner": l}
            for m in ("auc", "acc"): t = C.test(C.per_seed_json(j, m, l, ref)); r[m] = t["gain"]; r[m + " p"] = t["p"]
            rows.append(r)
        section("B3. Time-blocked screen (seeds 10800–10811): gain over af_base", T.to_md(pd.DataFrame(rows)))
    # C. confirmation 1
    if have("af_c1", "af_c1b"):
        dirs = str(RT / "af_c1"); blk = str(RT / "af_c1blocked") if have("af_c1blocked") else str(RT / "af_b1")
        out = C.run(spec, dirs, str(RT / "af_c1b"), blk); ver = C.verdicts(out, spec)
        n_seeds = out.n_seeds.max()
        section(f"C1. Confirmation 1 — verdicts (seeds 10830–10864, {int(n_seeds)} seeds with data)", T.to_md(ver, "{}"))
        section("C2. Confirmation 1 — all tests (candidates and controls vs af_base; Holm only over the candidate rows of the far and blocked families)", T.to_md(out))
        df = T.load(dirs, str(RT / "af_c1b"))
        section("C3. Confirmation seeds — absolute strata, af_base vs tuned graph-free comparator", T.to_md(T.absolute_table(df, [ref, spec["comparator"]])))
        for c in spec["candidates"]:
            section(f"C4. Dose-response of {c['name']} (gain over af_base by distance to the nearest typed anchor)", T.to_md(T.dose_table(df, c["name"], ref)))
        gp = RT / "af_jgroups_10830-10864.csv"
        if gp.exists():
            import af_power as P
            g = pd.read_csv(gp); dfg = df.merge(g, on=["seed", "split", "j"], how="left"); rows = []
            for c in spec["candidates"]:
                for split in "AB":
                    for m in ("auc", "acc"): rows.append({"candidate": c["name"], **P.two_way(dfg, c["name"], ref, split, spec["far"], 9.0, m, 200)})
            section("C5. Cluster-aware uncertainty of the far-stratum gains (two-way bootstrap over seeds and pair groups; MDE80 = 2.8 SE)", T.to_md(pd.DataFrame(rows)))
    else:
        section("C. Confirmation 1", "not available")


if __name__ == "__main__": main()

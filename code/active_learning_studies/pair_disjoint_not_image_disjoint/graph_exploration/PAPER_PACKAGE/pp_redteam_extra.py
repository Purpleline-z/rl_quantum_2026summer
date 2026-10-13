"""Other red-team audits from the stored files (development block 8800-8819, not confirmations): a3 time-blocked evaluation (hold out whole 100-frame blocks of a run, 25-frame buffer removed from training groups, pool and
graph) and a2 graph de-duplication, if their result directories exist.  Uses the red team's own gain_table; outputs tables/redteam_extra.csv/.md and numbers_redteam.json (keys rtx.<audit>.<cand>.<ref>.<metric>.<field>)."""
import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import RES, Numbers
import rt_tables as RT

PAIRS = [("lap3_k10_st4", "aw8_all_st2lr0.5"), ("lap3_k10", "aw8_all_st2lr0.5"), ("lap3_k10_st4", "lap3_k10"), ("lap3_k10_st4", "lap3_k10_st4_shuf")]


def main():
    N = Numbers("redteam"); rows = []
    for audit, cols in (("a3", ("split",)), ("a2", ("split", "delta"))):
        d = RES / "rt" / audit
        if not d.is_dir() or not any(d.glob("seed*.json")): continue
        df = RT.load_json_dir(d); nseeds = df.seed.nunique()
        for cand, ref in PAIRS:
            t = RT.gain_table(df, cand, ref, cols)
            if not len(t): continue
            for r in t.itertuples():
                key = "_".join(str(getattr(r, c)) for c in cols)
                for m in ("auc", "acc", "ll"):
                    if not hasattr(r, m): continue
                    rows.append(dict(audit=audit, cand=cand, ref=ref, group=key, metric=m, gain=getattr(r, m), lo=getattr(r, f"{m}_lo"), hi=getattr(r, f"{m}_hi"), p=getattr(r, f"{m}_p"), n_seeds=r.n_seeds))
                    for fld, v in (("gain", getattr(r, m)), ("lo", getattr(r, f"{m}_lo")), ("hi", getattr(r, f"{m}_hi")), ("p", getattr(r, f"{m}_p"))): N.put(f"rtx.{audit}.{key}.{cand}.{ref}.{m}.{fld}", float(v))
    t = pd.DataFrame(rows); t.to_csv(L.TABLES / "redteam_extra.csv", index=False)
    if len(t):
        show = t[t.metric.isin(["auc", "acc"])].copy(); show["gain [95% CI]"] = [L.fmt_gain(g, lo, hi) for g, lo, hi in zip(show.gain, show.lo, show.hi)]
        L.write(L.TABLES / "redteam_extra.md", "| audit | candidate - reference | group | metric | gain [95% CI] | p (uncorrected) | seeds |\n|---|---|---|---|---|---|---|\n" + "\n".join(f"| {r.audit} | {r.cand} - {r.ref} | {r.group} | {r.metric} | {r['gain [95% CI]']} | {L.fmt_p(r.p)} | {r.n_seeds} |" for _, r in show.iterrows()) + "\n")
    else: L.write(L.TABLES / "redteam_extra.md", "(no a2/a3 result files)\n")
    N.save(); print(len(t), "rows")


if __name__ == "__main__":
    main()

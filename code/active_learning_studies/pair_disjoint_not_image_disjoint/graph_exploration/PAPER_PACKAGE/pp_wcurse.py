"""Tuned-on-development caveat: for every frozen learner step, the gain on the development seeds (where the step was chosen, so optimistic) next to the gain on the pre-registered confirmation seeds
(paired over seeds, same label sets, AUC and accuracy, splits A and B).  The ratio confirmation/development shows how much of the development effect (winner's curse) survived.
Development learners of different directories are paired on the seeds they share (all learners of one seed use identical random label sets).
Outputs tables/winners_curse.csv, tables/winners_curse.md, numbers_wcurse.json (keys wc.<item>.<where>.<split>.<metric>.<field>)."""
import numpy as np
import pandas as pd

import pp_lib as L
from pp_lib import Numbers

R = "random"
ITEMS = [  # id, label, dev dirs, a, b, confirmations [(block label, dirs)]
    ("aw8", "aw8 vs default head", ["screen8_all"], "aw8", "baseline", [("ND1", ["confirm1"]), ("ND4", ["confirm4_htr"])]),
    ("aw8all", "aw8_all vs default head", ["screen8_all"], "aw8_all", "baseline", [("ND1", ["confirm1"]), ("ND4", ["confirm4_htr"])]),
    ("lap3", "lap3_k10 vs aw8_all", ["gl/dev1"], "lap3_k10", "aw8_all", [("GL1", ["gl/confirm1"]), ("GL2", ["gl/confirm2"])]),
    ("st4_vs_tuned", "lap3_k10_st4 vs aw8_all_st2lr0.5 (dev: 12 seeds)", ["sh/sdev5", "sh/sdev6"], "lap3_k10_st4", "aw8_all_st2lr0.5", [("SH1", ["sh/c1_learn"]), ("SH2", ["sh/c2_learn"])]),
    ("st4_vs_lap3", "lap3_k10_st4 vs lap3_k10", ["sh/sdev5", "sh/sdev1"], "lap3_k10_st4", "lap3_k10", [("SH1", ["sh/c1_learn"]), ("SH2", ["sh/c2_learn"])]),
]


def gains(dirs, a, b, split, metric):
    f = pd.concat([L.load_dir(d) for d in dirs], ignore_index=True); f = f[f.selector == R]; return L.per_seed(f, split, metric, (R, a), (R, b))


def main():
    N = Numbers("wcurse"); rows = []
    for iid, label, dev, a, b, confs in ITEMS:
        for split in ("A", "B"):
            for m in ("auc", "acc"):
                dv = L.summarize(gains(dev, a, b, split, m)); row = dict(item=label, split=split, metric=m, dev_gain=dv["gain"], dev_lo=dv["lo"], dev_hi=dv["hi"], dev_n=dv["n"])
                for fld in ("gain", "lo", "hi", "n"): N.put(f"wc.{iid}.dev.{split}.{m}.{fld}", dv[fld])
                for blk, dirs in confs:
                    c = L.summarize(gains(dirs, a, b, split, m)); row[f"{blk}_gain"] = c["gain"]; row[f"{blk}_lo"] = c["lo"]; row[f"{blk}_hi"] = c["hi"]; row[f"{blk}_ratio"] = c["gain"] / dv["gain"] if dv["gain"] else np.nan
                    for fld in ("gain", "lo", "hi", "n"): N.put(f"wc.{iid}.{blk}.{split}.{m}.{fld}", c[fld])
                    N.put(f"wc.{iid}.{blk}.{split}.{m}.ratio", row[f"{blk}_ratio"])
                rows.append(row)
    t = pd.DataFrame(rows); t.to_csv(L.TABLES / "winners_curse.csv", index=False)
    blocks = sorted({c[0] for i in ITEMS for c in i[5]}); lines = ["| learner step | split | metric | development (n seeds) | " + " | ".join(f"{b}: confirmation [95% CI] (confirmation/development)" for b in blocks) + " |", "|---|---|---|---|" + "---|" * len(blocks)]
    for _, r in t.iterrows():
        cells = []
        for b in blocks:
            cells.append(f"{L.fmt_gain(r[b + '_gain'], r[b + '_lo'], r[b + '_hi'])} ({r[b + '_ratio']:.2f})" if f"{b}_gain" in r and not pd.isna(r.get(f"{b}_gain")) else "")
        lines.append(f"| {r['item']} | {r.split} | {r.metric} | {L.fmt_gain(r.dev_gain, r.dev_lo, r.dev_hi)} (n={int(r.dev_n)}) | " + " | ".join(cells) + " |")
    L.write(L.TABLES / "winners_curse.md", "\n".join(lines) + "\n"); N.save()


if __name__ == "__main__":
    main()

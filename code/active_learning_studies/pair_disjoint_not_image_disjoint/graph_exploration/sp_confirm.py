"""Pre-registered confirmation analysis of the score-propagation study (rules frozen in PREREGISTRATION_SP1.md before any confirmation seed was run).

  python3 sp_confirm.py --spec SP1_spec.json --dirs DIR [DIR ...] [--blocked DIR] [--cold DIR] [--out FILE]

Per seed the gain of a candidate over the reference ``lap3_k10_st4`` is the mean over budgets and random label sets of the paired difference on identical label sets (log-loss: reference - candidate);
two-sided Wilcoxon over the seeds, paired bootstrap interval.  Families (Holm within each):
  main      candidates x splits {A, B} x {auc, acc}                two-sided superiority / inferiority test of the paired difference
  ni        the same 16 cells, one-sided non-inferiority tests      H0: median gain <= -margin (0.003 AUC, 0.01 accuracy); Holm within the family
  far       anchor-far stratum (far_auc, far_acc), two-sided        secondary
  htr       htr_acc, htr_ll, two-sided                              secondary (separate family)
  blocked   time-blocked contexts, auc / acc, two-sided             secondary
  cold      cold start (10 random initial judgments), auc / acc     secondary
Verdict per candidate on the main family: BETTER = all four gains > 0 and >= 3 of 4 Holm p < 0.05; WORSE = all four gains < 0 and >= 3 of 4 Holm p < 0.05; otherwise NON-INFERIOR if >= 3 of the 4 NI tests have Holm p < 0.05
(if the candidate is not BETTER); otherwise INCONCLUSIVE.  A candidate that is BETTER on `main` but whose `far` gains are not positive is reported as "gain only near anchors".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import rc_tables as T

MARGIN = {"auc": 0.003, "acc": 0.01}


def ni_p(per_seed: pd.Series, margin: float) -> float:
    x = per_seed.to_numpy() + margin
    try: return float(wilcoxon(x, alternative="greater").pvalue) if np.any(x != 0) else 1.0
    except ValueError: return 1.0


def family(df: pd.DataFrame, ref: str, cands: list[str], metrics, splits=("A", "B"), margins=None) -> pd.DataFrame:
    rows = []
    for c in cands:
        for split in splits:
            for m in metrics:
                ps = T.per_seed_gain(df, c, ref, split, m); s = T.summarize(ps); row = {"candidate": c, "split": split, "metric": m, **s}
                if margins is not None: row["p_ni"] = ni_p(ps, margins[m])
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["holm"] = T.holm(np.nan_to_num(out.p.to_numpy(), nan=1.0))
        if margins is not None: out["holm_ni"] = T.holm(np.nan_to_num(out.p_ni.to_numpy(), nan=1.0))
    return out


def verdicts(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in tab.groupby("candidate", sort=False):
        sig = int((g.holm < 0.05).sum()); ni = int((g.holm_ni < 0.05).sum()); pos, neg = bool((g.gain > 0).all()), bool((g.gain < 0).all())
        v = "BETTER" if pos and sig >= 3 else "WORSE" if neg and sig >= 3 else "NON-INFERIOR" if ni >= 3 else "INCONCLUSIVE"
        rows.append({"candidate": c, "all_gains_positive": pos, "all_gains_negative": neg, "holm<0.05 (of 4)": sig, "NI holm<0.05 (of 4)": ni, "verdict": v})
    return pd.DataFrame(rows)


BINS = ((-1, 9, "all judgments"), (-1, 0.001, "near (<=0.001)"), (0.001, 0.03, "0.001-0.03"), (0.03, 9, "far (>0.03)"))


def interpolation_share(dfd: pd.DataFrame, ref: str, ctl: str, fs: str) -> pd.DataFrame:
    """How much of the head's gain over the tuned graph-free head does the pure non-parametric model recover?  share = (fs - ctl) / (ref - ctl) in AUC (ratio of the seed means of the gains over the reference), bootstrap over seeds."""
    rows = []
    for split in ("A", "B"):
        for lo, hi, name in BINS:
            a = T.strata_gain(dfd, ctl, ref, split, lo, hi, "auc"); b = T.strata_gain(dfd, fs, ref, split, lo, hi, "auc"); both = pd.concat([a, b], axis=1, keys=["ctl", "fs"]).dropna()
            if len(both) < 5: continue
            x = both.to_numpy(); full = (x[:, 1].mean() - x[:, 0].mean()) / (-x[:, 0].mean()); rng = np.random.default_rng(0); idx = rng.integers(0, len(x), (2000, len(x)))
            bs = (x[idx, 1].mean(1) - x[idx, 0].mean(1)) / (-x[idx, 0].mean(1))
            rows.append({"split": split, "stratum": name, "n_seeds": len(x), "head - graph-free (AUC)": -x[:, 0].mean(), "head - free scores (AUC)": -x[:, 1].mean(), "share of the head's gain recovered by free scores": full,
                         "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
    return pd.DataFrame(rows)


def run(spec: dict, dirs, blocked=None, cold=None) -> str:
    ref, cands, controls = spec["reference"], spec["candidates"], spec.get("controls", []); df = T.load(dirs); out = []
    out.append("Seeds per learner/split:\n\n" + T.md(df.groupby(["learner", "split"]).agg(seeds=("seed", "nunique"), cells=("auc", "size")).reset_index(), fmt=".0f") + "\n")
    main = family(df, ref, cands, ("auc", "acc"), margins=MARGIN); v = verdicts(main)
    out.append(f"### Family `main` (two-sided, Holm over {len(main)} tests; gain = candidate - reference, positive = better; `p_ni` / `holm_ni` = non-inferiority tests, margins 0.003 AUC / 0.01 accuracy)\n\n{T.md(main)}\n\nVerdicts:\n\n{T.md(v)}\n")
    far = family(df, ref, cands, ("far_auc", "far_acc")); htr = family(df, ref, cands, ("htr_acc", "htr_ll"))
    out.append(f"### Family `far` (anchor-far stratum, secondary, Holm over {len(far)})\n\n{T.md(far)}\n")
    out.append(f"### Family `htr` (secondary, separate family, Holm over {len(htr)})\n\n{T.md(htr)}\n")
    summ = v.merge(far.groupby("candidate", sort=False).agg(far_all_pos=("gain", lambda x: bool((x > 0).all())), far_holm_sig=("holm", lambda x: int((x < 0.05).sum()))).reset_index(), on="candidate")
    summ["reading"] = [("better on the standard endpoint and positive in the anchor-far stratum" if r.verdict == "BETTER" and r.far_all_pos else
                        "better on the standard endpoint but NOT positive on anchor-far judgments: gain only near anchors" if r.verdict == "BETTER" else r.verdict.lower()) for r in summ.itertuples()]
    out.append("### Summary\n\n" + T.md(summ) + "\n")
    if controls: out.append("### Controls and baselines vs the reference (uncorrected, descriptive)\n\n" + T.md(family(df, ref, controls, ("auc", "acc", "far_auc", "far_acc", "ll"))) + "\n")
    out.append("### Per-type effects (gain over the reference per type, uncorrected, descriptive)\n\n" + T.md(T.types_table(df, ref, cands)) + "\n")
    dfd = T.load(dirs, keep_d=True)
    out.append("### Anchor strata (AUC and accuracy gain by the distance of the nearer test image to the nearest typed anchor; descriptive)\n\n" + T.md(T.strata_table(dfd, ref, cands + [c for c in controls if c != ref])) + "\n")
    ctl = next((c for c in controls if c.startswith("aw8_all")), None); fsn = next((c for c in cands if c.startswith("fs_")), None)
    if ctl and fsn: out.append("### How much of the head's gain over the tuned graph-free head is interpolation? (AUC, by anchor stratum; ratio of means, bootstrap over seeds; unstable where the denominator is near 0)\n\n" + T.md(interpolation_share(dfd, ref, ctl, fsn)) + "\n")
    if blocked:
        db = T.load(blocked); tb = family(db, ref, cands, ("auc", "acc"), splits=("blocked",)); out.append(f"### Family `blocked` (time-blocked split, secondary, Holm over {len(tb)})\n\n{T.md(tb)}\n")
        if controls: out.append("Controls on the blocked split:\n\n" + T.md(family(db, ref, controls, ("auc", "acc"), splits=("blocked",))) + "\n")
    if cold:
        dc = T.load(cold); tc = family(dc, ref, cands, ("auc", "acc")); out.append(f"### Family `cold` (cold start, secondary, Holm over {len(tc)})\n\n{T.md(tc)}\n")
        if controls: out.append("Controls in cold start:\n\n" + T.md(family(dc, ref, controls, ("auc", "acc"))) + "\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--spec", type=Path, required=True); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--blocked", nargs="*"); ap.add_argument("--cold", nargs="*")
    ap.add_argument("--out", type=Path); a = ap.parse_args(); fix = lambda ds: [str((T.NEW / d) if not Path(d).exists() else d) for d in ds] if ds else None
    text = run(json.loads(a.spec.read_text()), fix(a.dirs), fix(a.blocked), fix(a.cold))
    if a.out: a.out.write_text(text)
    else: print(text)


if __name__ == "__main__": main()

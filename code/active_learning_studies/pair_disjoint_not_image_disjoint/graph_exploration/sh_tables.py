"""Tables of the selection x graph-learner x HTR study, generated from the result JSONs (nothing typed by hand).

  python3 sh_tables.py dev DIR [DIR ...] [--learner=lap3_k10]         gain of every selector over RANDOM selection with the same final learner (paired by seed, split, budget; random = mean of 3 draws)
  python3 sh_tables.py comp DIR [DIR ...] [--learner=lap3_k10]        mean type composition (judgments per type) of the chosen sets, by budget
  python3 sh_tables.py cross DIR [...]                                every (selector, learner) cell against random selection with aw8_all (the previous best learner)
  python3 sh_tables.py confirm SPEC.json DIR [DIR ...]                the pre-registered tests and verdicts of a spec (see PREREGISTRATION_SH1.md)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

NEW = Path(__file__).resolve().parents[2] / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(NEW.parents[1] / "code" / "active_learning_program"))
import nd_confirm as c  # noqa: E402

METRICS = ("auc", "acc", "ll", "htr_acc", "htr_ll", "t1x1_acc", "tc6x2_acc", "t13_acc")


def load(dirs): return c.load(dirs)


def summarize(per):
    rng = np.random.default_rng(0); b = rng.choice(per.values, (3000, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue) if np.any(per.values != 0) else 1.0
    except ValueError: p = 1.0
    return {"gain": per.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5), "share": float((per > 0).mean()), "p": p, "n": len(per)}


def gain(frame, split, metric, a, b):
    sign = c.SIGN[metric]; g = frame[frame.split == split]
    mean = lambda cell: g[(g.selector == cell[0]) & (g.learner == cell[1])].groupby(["seed", "budget"])[metric].mean()
    return (sign * (mean(a) - mean(b))).dropna().groupby(level="seed").mean()


def dev_table(frame, learner, ref_sel="random", metrics=("auc", "acc", "ll", "htr_acc", "htr_ll")):
    rows = []
    for split in sorted(frame.split.unique()):
        sels = sorted(set(frame[(frame.split == split) & (frame.learner == learner)].selector) - {ref_sel})
        for s in sels:
            r = {"split": split, "selector": s}
            for m in metrics:
                try: o = summarize(gain(frame, split, m, (s, learner), (ref_sel, learner))); r[m] = o["gain"]; r[m + "_p"] = o["p"]; r["n"] = o["n"]
                except Exception: pass
            rows.append(r)
    out = pd.DataFrame(rows)
    for m in metrics:
        if m + "_p" in out:
            for split in out.split.unique(): k = out.split == split; out.loc[k, m + "_holm"] = c.holm(out.loc[k, m + "_p"])
    return out


def comp_table(frame, learner):
    f = frame[frame.learner == learner]; cols = [x for x in ("n_type0", "n_type2", "n_type3", "n_type4") if x in f]
    t = f.groupby(["selector", "budget"])[cols].mean().round(1).reset_index(); t.columns = ["selector", "budget", "(1x1)", "c(6x2)", "RT13", "HTR"][:len(t.columns)]; return t


def to_md(df, floatfmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def build_tests(spec):
    """Main family (Holm): candidate selector vs random selection under the spec's learner, splits x (auc, acc); control rows (uncorrected): candidate vs its own control (shuffled graph / plain vopt_u)
    on the same metrics; HTR family (Holm within family): htr_acc and htr_ll; secondary (exploratory): ll and the three other types' accuracy."""
    L, T = spec["learner"], []
    for cand in spec["candidates"]:
        for split in spec["splits"]:
            for m in ("auc", "acc"): T.append({"family": "main", "kind": "vs_random", "cand": cand, "split": split, "metric": m, "a": [cand, L], "b": ["random", L]})
            for name, ctl in spec["controls"].get(cand, {}).items():
                for m in ("auc", "acc"): T.append({"family": "control:" + name, "kind": "vs_ctl:" + name, "cand": cand, "split": split, "metric": m, "a": [cand, L], "b": [ctl, L]})
            for m in ("htr_acc", "htr_ll"): T.append({"family": "htr", "kind": "vs_random", "cand": cand, "split": split, "metric": m, "a": [cand, L], "b": ["random", L]})
            for m in ("ll", "t1x1_acc", "tc6x2_acc", "t13_acc"): T.append({"family": "secondary:" + m, "kind": "vs_random", "cand": cand, "split": split, "metric": m, "a": [cand, L], "b": ["random", L]})
    return {"tests": T}


def verdicts(out, spec):
    rows = []
    for cand in spec["candidates"]:
        main = out[(out.family == "main") & (out.cand == cand)]; ctl = out[(out.kind == "vs_ctl:shuf") & (out.cand == cand)]; htr = out[(out.family == "htr") & (out.cand == cand)]
        rest = out[(out.family.str.startswith("secondary:t")) & (out.cand == cand)]
        row = {"candidate": cand, "all_four_positive": bool(len(main) == 4 and (main.gain > 0).all()), "n_holm_sig": int((main.holm < 0.05).sum()), "beats_control": bool(((ctl.gain > 0) & (ctl.p < 0.05)).any()) if len(ctl) else None}
        row["MAIN_CONFIRMED"] = bool(row["all_four_positive"] and row["n_holm_sig"] >= 3 and row["beats_control"] in (True, None))
        row["htr_acc_holm_sig_splits"] = int(((htr.metric == "htr_acc") & (htr.holm < 0.05) & (htr.gain > 0)).sum()); row["htr_ll_holm_sig_splits"] = int(((htr.metric == "htr_ll") & (htr.holm < 0.05) & (htr.gain > 0)).sum())
        row["other_type_acc_worst_gain"] = float(rest.gain.min()) if len(rest) else float("nan"); rows.append(row)
    return pd.DataFrame(rows)


def confirm_report(spec, dirs):
    out = c.run(build_tests(spec), load(dirs)); out["cells"] = [f"{a[0]} - {b[0]}" for a, b in zip(out.a, out.b)]
    cols = ["family", "split", "metric", "cells", "n_seeds", "gain", "lo", "hi", "share_better", "p", "holm"]
    return out, to_md(out[cols].assign(**{k: out[k].astype(float) for k in ("gain", "lo", "hi", "share_better", "p", "holm")}))


if __name__ == "__main__":
    mode = sys.argv[1]; dirs = [a for a in sys.argv[2:] if not a.startswith("--") and not a.endswith(".json")]; opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    if mode == "dev": print(to_md(dev_table(load(dirs), opts.get("learner", "lap3_k10"))))
    elif mode == "comp": print(to_md(comp_table(load(dirs), opts.get("learner", "lap3_k10")), "{:.1f}"))
    elif mode == "cross":
        f = load(dirs); rows = []
        for split in sorted(f.split.unique()):
            for (s, l), _ in f[f.split == split].groupby(["selector", "learner"]):
                r = {"split": split, "selector": s, "learner": l}
                for m in ("auc", "acc", "ll", "htr_acc"): o = summarize(gain(f, split, m, (s, l), ("random", "aw8_all"))); r[m] = o["gain"]; r[m + "_p"] = o["p"]
                rows.append(r)
        print(to_md(pd.DataFrame(rows)))
    elif mode == "confirm":
        spec = json.loads(Path([a for a in sys.argv[2:] if a.endswith(".json")][0]).read_text()); out, md = confirm_report(spec, dirs); print(md); print(); print(to_md(verdicts(out, spec), "{}"))

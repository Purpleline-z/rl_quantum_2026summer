"""Tables of the graph-learner study, generated from the result JSONs (nothing typed by hand).

  python3 gl_tables.py dev results/gl/dev1 [--ref aw8_all] [--out FILE]      development table: gain of every learner over the reference, per split, with control columns
  python3 gl_tables.py confirm SPEC.json results/gl/confirm1 [--out FILE]    pre-registered tests (same machinery as nd_confirm.py, plus paired candidate-minus-control rows)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

NEW = Path(__file__).resolve().parents[2] / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(NEW.parents[1] / "code" / "active_learning_program"))
import nd_confirm as c  # noqa: E402

CONTROLS = {"_shuf": "shuffled graph", "_noprop": "no propagation", "cons": "no-graph consistency"}


def load(dirs):
    f = c.load(dirs); f["selector"] = f["selector"] if "selector" in f else "random"; return f


def gain(frame, split, metric, a, b, selector="random"):
    per = c.gains(frame, split, metric, (selector, a), (selector, b)); return per


def summarize(per):
    rng = np.random.default_rng(0); b = rng.choice(per.values, (3000, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue) if np.any(per.values != 0) else 1.0
    except ValueError: p = 1.0
    return {"gain": per.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5), "share": float((per > 0).mean()), "p": p, "n": len(per)}


def control_of(name: str):
    if name.endswith("_shuf"): return None
    if name.startswith("rq"): return name + "_shuf"
    if name.startswith(("lap", "sgc", "gcn2")): return name + "_shuf"
    return None


def dev_table(frame, ref="aw8_all", metrics=("auc", "acc", "ll", "htr_acc", "htr_ll")):
    rows = []
    for split in sorted(frame.split.unique()):
        learners = sorted(set(frame[frame.split == split].learner) - {ref})
        for l in learners:
            r = {"split": split, "learner": l}
            for m in metrics:
                try: s = summarize(gain(frame, split, m, l, ref)); r[m] = s["gain"]; r[m + "_lo"] = s["lo"]; r[m + "_hi"] = s["hi"]; r["n"] = s["n"]
                except Exception: pass
            ctl = control_of(l)
            if ctl and ctl in set(frame.learner):
                for m in ("auc", "acc"):
                    s = summarize(gain(frame, split, m, l, ctl)); r[m + "_vs_ctl"] = s["gain"]; r[m + "_vs_ctl_p"] = s["p"]
            rows.append(r)
    return pd.DataFrame(rows)


def to_md(df, floatfmt="{:+.4f}"):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def build_tests(spec):
    """Tests of a G-spec: main family (candidate - reference, A/B x AUC/accuracy; Holm), control rows (candidate - own shuffled-graph control, AUC/accuracy; uncorrected, plus any extra
    controls listed), secondary exploratory families (log-loss and HTR metrics; Holm within family, not part of the success rule)."""
    ref, T = spec["reference"], []
    cell = lambda l: ["random", l]
    for cand in spec["candidates"]:
        for split in spec["splits"]:
            for m in ("auc", "acc"): T.append({"family": "main", "kind": "vs_ref", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
            for ctl_name, ctl in spec["controls"].get(cand, {}).items():
                for m in ("auc", "acc"): T.append({"family": "control:" + ctl_name, "kind": "vs_ctl:" + ctl_name, "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ctl)})
            for m in spec.get("secondary_metrics", ["ll", "htr_acc", "htr_ll"]): T.append({"family": "secondary:" + m, "kind": "vs_ref", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
    return {"tests": T}


def verdicts(out, spec):
    """Success rule of the pre-registration, applied to the test table: per candidate, (i) all four reference differences positive, (ii) >= 3 of them with Holm p < 0.05,
    (iii) beats its shuffled-graph control (positive and uncorrected p < 0.05 for AUC or accuracy in >= 1 split)."""
    rows = []
    for cand in spec["candidates"]:
        main = out[(out.family == "main") & (out.cand == cand)]; ctl = out[(out.kind == "vs_ctl:shuf") & (out.cand == cand)]
        i = bool((main.gain > 0).all() and len(main) == 4); ii = int((main.holm < 0.05).sum()); iii = bool(((ctl.gain > 0) & (ctl.p < 0.05)).any())
        rows.append({"candidate": cand, "all_four_positive": i, "n_holm_sig": ii, "beats_shuffled": iii, "CONFIRMED": bool(i and ii >= 3 and iii)})
    return pd.DataFrame(rows)


def confirm_report(spec, dirs):
    out = c.run(build_tests(spec), load(dirs)); out["cells"] = [f"{a[1]} - {b[1]}" for a, b in zip(out.a, out.b)]
    cols = ["family", "split", "metric", "cells", "n_seeds", "gain", "lo", "hi", "share_better", "p", "holm"]
    return out, to_md(out[cols].assign(**{k: out[k].astype(float) for k in ("gain", "lo", "hi", "share_better", "p", "holm")}), "{:+.4f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        ref = "aw8_all"; dirs = [a for a in sys.argv[2:] if not a.startswith("--")]; frame = load(dirs); t = dev_table(frame, ref)
        t = t[[c_ for c_ in t.columns if not c_.endswith("_lo") and not c_.endswith("_hi")]]
        print(to_md(t))
    elif mode == "confirm":
        spec = json.loads(Path(sys.argv[2]).read_text()); out, md = confirm_report(spec, sys.argv[3:]); print(md); print(); print(to_md(verdicts(out, spec), "{}"))

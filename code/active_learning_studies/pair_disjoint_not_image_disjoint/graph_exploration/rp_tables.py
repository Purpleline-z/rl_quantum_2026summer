"""Tables of the representation study (generated from result JSONs; nothing typed by hand).

  python3 rp_tables.py dev DIR [DIR..] [--refs=lap3_k10_st4,aw8_all_st2lr0.5] [--only=a,b]   gain of every learner over each reference (AUC, accuracy, log-loss, HTR accuracy / log-loss), per split
  python3 rp_tables.py confirm SPEC.json DIR [DIR..]                                           pre-registered tests of PREREGISTRATION_R1 (see spec)
  python3 rp_tables.py diff DIR.. --pairs=a:b,c:d                                              exploratory paired differences incl. per type (uncorrected p)
  python3 rp_tables.py types DIR.. --learners=a,b [--ref=lap3_k10_st4]  /  budget ...          per-type / per-budget gains
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
NEW = HERE.parents[1] / "new_directions"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(NEW)); sys.path.insert(0, str(NEW.parents[1] / "active_learning_program"))
import gl_tables as gt  # noqa: E402
import nd_confirm as c  # noqa: E402


def load_complete(ds):
    """Like gl_tables.load, but skips cell files that have fewer records than the fullest file of their directory (unfinished or killed runs)."""
    rows = []
    for d in ds:
        files = sorted(Path(d).glob("seed*_*.json")); recs = {f: json.loads(f.read_text()) for f in files}; full = max((len(v) for v in recs.values()), default=0)
        rows += [r for f, v in recs.items() if len(v) >= full for r in v]
    f = pd.DataFrame(rows); f["selector"] = "random"; return f


def arg(prefix, default=None):
    for a in sys.argv[2:]:
        if a.startswith(prefix + "="): return a.split("=", 1)[1]
    return default


def dirs():
    return [a for a in sys.argv[2:] if not a.startswith("--") and not a.endswith(".json")]


def confirm_tables(spec, frame):
    """Main family: candidate - reference (A/B x AUC/accuracy), Holm over all candidates; controls (shuffled graph, capacity), tuned graph-free control and no-op references uncorrected; HTR secondary family (own Holm)."""
    ref, T = spec["reference"], []; cell = lambda l: ["random", l]; have = set(frame.learner)
    for cand, ctl in spec["candidates"].items():
        for split in spec["splits"]:
            for m in ("auc", "acc"):
                T.append({"family": "main", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
                for k, name in ctl.items():
                    if name in have: T.append({"family": f"control:{k}", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(name)})
                T.append({"family": "vs_graphfree", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(spec["graphfree"])})
            for m in ("ll", "htr_acc", "htr_ll"): T.append({"family": f"secondary:{m}", "cand": cand, "split": split, "metric": m, "a": cell(cand), "b": cell(ref)})
    for split in spec["splits"]:
        for m in ("auc", "acc"): T.append({"family": "baseline_vs_graphfree", "cand": ref, "split": split, "metric": m, "a": cell(ref), "b": cell(spec["graphfree"])})
    out = c.run({"tests": T}, frame.assign(selector="random")); out["cells"] = [f"{a[1]} - {b[1]}" for a, b in zip(out.a, out.b)]
    rows = []
    for cand in spec["candidates"]:
        main = out[(out.family == "main") & (out.cand == cand)]; allpos = bool((main.gain > 0).all() and len(main) == 4); nsig = int(((main.holm < 0.05) & (main.gain > 0)).sum()); ctl_ok = {}
        for k in spec["candidates"][cand]:
            cc = out[(out.family == f"control:{k}") & (out.cand == cand)]; ctl_ok[k] = bool(((cc.gain > 0) & (cc.p < 0.05)).any()) if len(cc) else False
        rows.append({"candidate": cand, "all_four_positive": allpos, "n_holm_sig": nsig, **{f"beats_{k}": v for k, v in ctl_ok.items()}, "CONFIRMED": bool(allpos and nsig >= 3 and all(ctl_ok.values()))})
    return out, pd.DataFrame(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        frame = load_complete(dirs()); only = arg("--only"); refs = (arg("--refs") or "lap3_k10_st4").split(",")
        for ref in refs:
            t = gt.dev_table(frame, ref, metrics=("auc", "acc", "ll", "htr_acc", "htr_ll")); t = t[[x for x in t.columns if not x.endswith(("_lo", "_hi", "_vs_ctl", "_vs_ctl_p"))]]
            if only: t = t[t.learner.isin(only.split(","))]
            print(f"\n#### gain over `{ref}` (AUC/accuracy: learner - ref; log-loss: ref - learner; mean over budgets, draws, then seeds)\n"); print(gt.to_md(t))
    elif mode == "confirm":
        spec = json.loads(Path(sys.argv[2]).read_text()); frame = load_complete([a for a in sys.argv[3:] if not a.startswith("--")]); out, verdict = confirm_tables(spec, frame)
        cols = ["family", "split", "metric", "cells", "n_seeds", "gain", "lo", "hi", "share_better", "p", "holm"]; print(gt.to_md(out[cols].assign(**{k: out[k].astype(float) for k in ("gain", "lo", "hi", "share_better", "p", "holm")}))); print(); print(gt.to_md(verdict, "{}"))
    elif mode == "diff":
        frame = load_complete(dirs()); rows = []
        for pr in arg("--pairs").split(","):
            a, b = pr.split(":")
            for split in sorted(frame.split.unique()):
                for m in ("auc", "acc", "ll", "htr_acc", "htr_ll", "t1x1_acc", "tc6x2_acc", "t13_acc"):
                    try: s = gt.summarize(gt.gain(frame, split, m, a, b)); rows.append({"pair": f"{a} - {b}", "split": split, "metric": m, "gain": s["gain"], "lo": s["lo"], "hi": s["hi"], "share_better": s["share"], "p_uncorrected": s["p"], "n": s["n"]})
                    except Exception: pass
        print(gt.to_md(pd.DataFrame(rows)))
    elif mode in ("budget", "types"):
        cands = arg("--learners").split(","); frame = load_complete(dirs()); ref = arg("--ref", "lap3_k10_st4")
        print(gt.to_md(gt.budget_table(frame, cands, ref) if mode == "budget" else gt.type_table(frame, cands, ref)))

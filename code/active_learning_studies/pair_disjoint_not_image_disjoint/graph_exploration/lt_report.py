"""Builds RESULTS_latent_type.md from RESULTS_latent_type.template.md and the result directories (every table is generated here; nothing typed by hand).
  python3 lt_report.py [--dev-only]"""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import lt_tables as T, rt_tables as rt

HERE = Path(__file__).resolve().parent; RES = HERE.parents[1] / "new_directions" / "results" / "lt"
BASE = "lap3_k10_st4"; CANDS = ["lt_aux0.5_em", "lt_aux0.1_em", "lt_lik_em", "lt_pa0.25_em"]
CONTROL = {"lt_aux0.5_em": "lt_aux0.5_prior", "lt_aux0.1_em": "lt_aux0.1_prior", "lt_lik_em": "lt_lik_prior", "lt_pa0.25_em": "lt_pa0.25_prior"}
EXTRA = {"lt_aux0.5_em": ["lt_aux0.5_random", "lt_aux0.5_shufprior"]}


def fmt(v, p=False):
    if isinstance(v, float) and np.isnan(v): return ""
    return f"{v:.3f}" if p else f"{v:+.4f}"


def md(rows, cols, heads=None):
    heads = heads or cols; out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in rows: out.append("| " + " | ".join(str(r[c]) if isinstance(r[c], str) or not isinstance(r[c], float) else fmt(r[c], c.endswith(("_p", " p", "_holm", "share"))) for c in cols) + " |")
    return "\n".join(out)


def gains(cdir, adir, cands, ref=BASE, thr=0.01):
    df = T.load(RES / cdir); long = T.load_long(RES / cdir) if adir else None; anc = rt.load_a1b(RES / adir) if adir else None
    return T.table(df, long, anc, cands, ref, thr=thr), df


def add_holm(t, cands):
    t = t.copy()
    for stratum in t.stratum.unique():
        sel = t[(t.stratum == stratum) & t.cand.isin(cands)]
        keys = [(i, m) for i in sel.index for m in ("auc", "acc")]; adj = T.holm([t.loc[i, f"{m}_p"] for i, m in keys])
        for (i, m), a in zip(keys, adj): t.loc[i, f"{m}_holm"] = a
    return t


def gains_md(t):
    rows = [{**r, "gain": f"{fmt(r['auc'])} [{fmt(r['auc_lo'])},{fmt(r['auc_hi'])}]", "gacc": f"{fmt(r['acc'])} [{fmt(r['acc_lo'])},{fmt(r['acc_hi'])}]", "gll": fmt(r["ll"])} for r in t.to_dict("records")]
    cols = ["cand", "split", "stratum", "gain", "auc_p", "auc_holm", "gacc", "acc_p", "acc_holm", "gll", "auc_share", "n_seeds"] if "auc_holm" in t else ["cand", "split", "stratum", "gain", "auc_p", "gacc", "acc_p", "gll", "auc_share", "n_seeds"]
    heads = {"gain": "AUC gain [95% CI]", "gacc": "acc gain [95% CI]", "gll": "log-loss gain", "auc_p": "AUC p (Wilcoxon)", "acc_p": "acc p", "auc_holm": "AUC Holm p", "acc_holm": "acc Holm p", "auc_share": "share seeds AUC>0"}
    return md(rows, cols, [heads.get(c, c) for c in cols])


def verdict(cdir, adir):
    t, df = gains(cdir, adir, CANDS); t = add_holm(t, CANDS); rows = []
    for c in CANDS:
        s = t[(t.cand == c) & (t.stratum == "all")]; f = t[(t.cand == c) & (t.stratum.str.startswith("far"))]
        std_pos = all(s[m].gt(0).all() for m in ("auc", "acc")) and len(s) == 2; std_n = int((s[["auc_holm", "acc_holm"]] < 0.05).to_numpy().sum())
        far_pos = all(f[m].gt(0).all() for m in ("auc", "acc")) and len(f) == 2; far_n = int((f[["auc_holm", "acc_holm"]] < 0.05).to_numpy().sum())
        ctl = T.table(df, None, None, [c], CONTROL[c]); ctl_pos = all(ctl[m].gt(0).all() for m in ("auc", "acc")) and len(ctl) == 2
        extra = {x: (lambda tt: all(tt[m].gt(0).all() for m in ("auc", "acc")))(T.table(df, None, None, [c], x)) for x in EXTRA.get(c, [])}
        a = std_pos and std_n >= 3; b = far_pos; cc = ctl_pos and all(extra.values())
        rows.append({"cand": c, "(a) std: 4 gains > 0": "yes" if std_pos else "no", "std Holm<0.05 (of 4)": std_n, "(b) far: 4 gains > 0": "yes" if far_pos else "no", "far Holm<0.05 (of 4)": far_n, "prior-only control": CONTROL[c], "(c) beats it (4 gains > 0)": "yes" if ctl_pos else "no",
                     "other controls": ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in extra.items()) or "-", "success (a and b and c)": "YES" if (a and b and cc) else "no"})
    return t, rows


def controls_md(cdir):
    df = T.load(RES / cdir); out = []
    for c in CANDS:
        t = T.table(df, None, None, [c], CONTROL[c])
        for x in EXTRA.get(c, []): t = pd.concat([t, T.table(df, None, None, [c], x).assign(cand=f"{c} - {x}")])
        t["cand"] = [f"{c} − {CONTROL[c]}" if n == c else n for n in t.cand]; out.append(t)
    return gains_md(pd.concat(out))


def diag_md():
    rows = [r for f in glob.glob(str(RES / "diag_dev" / "*.json")) for r in json.load(open(f))]; df = pd.DataFrame(rows); out = []
    for b, g in df.groupby("budget"):
        out.append({"budget": b, "prior acc": g.prior_acc.mean(), "prior+outcomes (EM) acc": g.em_acc.mean(), "random posteriors acc": g.random_acc.mean(), "shuffled prior acc": g.shuf_prior_acc.mean(), "prior LL": g.prior_ll.mean(), "EM LL": g.em_ll.mean(), "random LL": g.random_ll.mean(),
                    "near (≤0.01) prior→EM": f"{g.prior_acc_near.mean():.3f}→{g.em_acc_near.mean():.3f}", "far prior→EM": f"{g.prior_acc_far.mean():.3f}→{g.em_acc_far.mean():.3f}"})
    return md([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in out], list(out[0]))


def dev_md(name, cands):
    t, _ = gains(name, "a1b_dev", cands); return gains_md(t)


def pooled_md():
    out = []
    for c in CANDS:
        for split in ("A", "B"):
            vals = {}
            for m in ("auc", "acc"):
                per = pd.concat([T.seed_gains_standard(T.load(RES / d), c, BASE, split)[m] for d in ("conf1", "conf2") if (RES / d).exists()]); vals[m] = T.summarise(per)
            out.append({"cand": c, "split": split, "seeds": vals["auc"]["n"], "AUC gain": f"{fmt(vals['auc']['gain'])} [{fmt(vals['auc']['lo'])},{fmt(vals['auc']['hi'])}]", "AUC p": vals["auc"]["p"], "acc gain": f"{fmt(vals['acc']['gain'])} [{fmt(vals['acc']['lo'])},{fmt(vals['acc']['hi'])}]", "acc p": vals["acc"]["p"]})
    return md(out, ["cand", "split", "seeds", "AUC gain", "AUC p", "acc gain", "acc p"], ["cand", "split", "seeds", "AUC gain [95% CI]", "AUC p", "acc gain [95% CI]", "acc p"])


def blocked_md(d):
    df = T.load(RES / d); rows = []
    for c in CANDS + ["aw8_all_st2lr0.5", "lt_aux0.5_prior"]:
        a = df[df.learner == c].set_index(["seed", "budget", "draw"]); b = df[df.learner == BASE].set_index(["seed", "budget", "draw"]); common = a.index.intersection(b.index)
        if not len(common): continue
        r = {"cand": c, "n_seeds": len({i[0] for i in common})}
        for m in ("auc", "acc"): s = T.summarise((a.loc[common, m] - b.loc[common, m]).groupby(level=0).mean()); r[m] = f"{fmt(s['gain'])} [{fmt(s['lo'])},{fmt(s['hi'])}]"; r[m + "_p"] = s["p"]
        rows.append(r)
    return md(rows, ["cand", "n_seeds", "auc", "auc_p", "acc", "acc_p"], ["arm − lap3_k10_st4", "seeds", "AUC gain [95% CI]", "AUC p", "acc gain [95% CI]", "acc p"])


def bins_md(c):
    long = T.load_long(RES / c); anc = rt.load_a1b(RES / f"a1b_{c}"); rows = []
    for cand in ("lt_aux0.5_em", "lt_lik_em"):
        t = rt.auc_by_stratum(long, anc, cand, BASE, [("<=0.001", -1, 0.001), ("0.001-0.01", 0.001, 0.01), ("0.01-0.03", 0.01, 0.03), (">0.03", 0.03, 9)], "anc_all_min"); t.insert(0, "cand", cand); rows.append(t)
    t = pd.concat(rows); return md(t.to_dict("records"), ["cand", "split", "stratum", "n_judgments", "n_seeds", "auc_gain", "CI"], ["cand", "split", "distance to nearest typed anchor", "judgments (all seeds, budgets, 1 draw)", "seeds", "AUC gain", "95% CI"])


def section(key):
    ready = lambda d: (RES / d).exists() and len(list((RES / d).glob("*.json"))) > 0
    if key == "diag": return diag_md()
    if key == "dev1": return dev_md("dev1", ["aw8_all_st2lr0.5", "lt_pa0.25_em", "lt_pa1_em", "lt_pa1_prior", "lt_lik_em", "lt_lik_prior", "lt_aux0.5_em", "lt_aux0.5_prior"])
    if key == "dev2": return dev_md("dev2", ["lt_aux0.1_em", "lt_aux1_em", "lt_aux0.5_lik_em", "lt_aux0.5_random", "lt_pa0.5far_em", "lt_pa0.5thr0.8_em", "lt_pa0.5far_thr0.8_em"])
    if key == "dev3": return dev_md("dev3", [f"lt_inf{l}_{p}" for l in ("0.5", "1", "2") for p in ("em", "prior", "shufprior", "random")])
    if key == "dev4": return dev_md("dev4", ["lt_auxall0.1_em", "lt_auxall0.5_em", "lt_auxall0.1_prior", "lt_auxall0.5_prior", "lt_auxall0.5_random", "lt_aux0.5_shufprior"])
    m = re.match(r"(conf[12])_(verdict|gains|controls|blocked|graphfree)$", key)
    if m:
        c, kind = m.groups()
        if kind == "blocked": return blocked_md(f"{c}_blocked") if ready(f"{c}_blocked") else "(not run)"
        if not ready(c): return "(not run)"
        if kind == "verdict": t, rows = verdict(c, f"a1b_{c}"); return md(rows, list(rows[0]))
        if kind == "gains": t, _ = verdict(c, f"a1b_{c}"); return gains_md(t)
        if kind == "controls": return controls_md(c)
        if kind == "graphfree": t, _ = gains(c, f"a1b_{c}", ["aw8_all_st2lr0.5"]); return gains_md(t)
    if key in ("conf1_bins", "conf2_bins"): return bins_md(key[:5]) if ready(key[:5]) and (RES / f"a1b_{key[:5]}").exists() else "(not run)"
    if key == "pooled": return pooled_md() if ready("conf1") and ready("conf2") else "(not run)"
    raise KeyError(key)


if __name__ == "__main__":
    tpl = (HERE / "RESULTS_latent_type.template.md").read_text(); out = re.sub(r"<!--TABLE:(\w+)-->", lambda m: section(m.group(1)), tpl)
    out = re.sub(r"<!--SECTION:(\w+)-->", lambda m: (HERE / f"results_narrative_{m.group(1)}.md").read_text() if (HERE / f"results_narrative_{m.group(1)}.md").exists() else "", out)
    (HERE / "RESULTS_latent_type.md").write_text(out); print("written")

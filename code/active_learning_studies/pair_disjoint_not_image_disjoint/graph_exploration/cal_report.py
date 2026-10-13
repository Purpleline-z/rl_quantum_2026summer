"""Builds RESULTS_calibration.md: the hand-written narrative (results_narrative_calibration.md) with every {{TABLE:name}} replaced by the output of `new_directions/cal_tables.py` / `cal_figures.py`
run on the stored cells (nothing typed by hand).  python3 cal_report.py      (run from this directory; tables whose result directory does not exist are marked 'not run')"""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ND = HERE.parents[1] / "new_directions"; RES = ND / "results" / "cal"; SPEC = str(HERE / "CAL1_spec.json")
NOISE = re.compile(r"warn|Validated|Loaded SimCLR|self\.ahat|xa, xb|Excluding", re.I)
DEV_V = "T_glob,T_type_tau8,T_type0_tau2,T_typeprior_tau8,T_typeprior_fixed,CF,CF_typeprior_tau8,T_conf_anc_min,shrink_far,Davidson,ENS,ENS_T,ls0.1,ls0.1_T,ls0.25,ls0.25_T"
TABLES = {
    "dev1_gain": ["dev", "results/cal/dev1", "--only=gain", f"--variants={DEV_V}"], "dev2_gain": ["dev", "results/cal/dev2", "--only=gain", "--variants=T_glob,T_type0_tau2,T_typeprior_tau8,CF,CF_typeprior_tau8"],
    "dev1_levels": ["dev", "results/cal/dev1", "--only=levels", "--variants=T_type0_tau2,T_typeprior_tau8,CF,CF_typeprior_tau8"], "oracle": ["oracle", "results/cal/dev1"], "budget_dev": ["budget", "results/cal/dev2", "--variants=T_type0_tau2,T_typeprior_tau8,CF_typeprior_tau8"],
    "learners_dev": ["learners", str(HERE / "CAL_dev_learners_spec.json")], "learners_dev_lap3k10": ["learners", str(HERE / "CAL_dev_learners_vs_lap3k10_spec.json")], "rt13off_dev": ["learners", str(HERE / "CAL_dev_rt13off_spec.json")], "kfold_dev": ["kfold", SPEC, "results/cal/dev_k3", "results/cal/dev1", "results/cal/dev_k10"],
    "novelty": ["novelty", SPEC, "results/cal/conf1"], "sensitivity": ["sensitivity", SPEC, "results/cal/conf1"], "transfer_dev": ["transfer", "results/cal/dev1", "results/cal/dev2"], "confirm": ["confirm", SPEC, "results/cal/conf1"], "conf_explore": ["dev", "results/cal/conf1", "--only=gain", "--variants=T_glob,T_type_tau8,T_type0_tau0.5,T_typeprior_tau2,T_typeprior_fixed,CF_typeprior_fixed,T_conf_anc_min,shrink_far,Davidson,CF_T"], "conf_budget": ["budget", "results/cal/conf1", "--variants=T_type0_tau2,T_typeprior_tau8,CF,CF_typeprior_tau8"], "learners_conf": ["learners", str(HERE / "CAL1_learners_spec.json")],
}


def run(args) -> str:
    dirs = [a for a in args[1:] if a.startswith("results/")]
    if dirs and not all((ND / d).is_dir() and any((ND / d).glob("seed*.pkl.gz")) for d in dirs): return "*not run*"
    out = subprocess.run([sys.executable, "cal_tables.py", *args], cwd=ND, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ND.parents[1] / "active_learning_program")})
    if out.returncode: return f"*failed*: {out.stderr[-300:]}"
    return "\n".join(l for l in out.stdout.splitlines() if not NOISE.search(l))


def main():
    text = (HERE / "results_narrative_calibration.md").read_text()
    def sub(m):
        name = m.group(1)
        if name not in TABLES: return f"*unknown table {name}*"
        return run(TABLES[name])
    (HERE / "RESULTS_calibration.md").write_text(re.sub(r"\{\{TABLE:(\w+)\}\}", sub, text)); print("wrote RESULTS_calibration.md")


if __name__ == "__main__": main()

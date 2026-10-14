"""Builds RESULTS_graph_gp.md: the hand-written narrative (results_narrative_graph_gp.md) with every {{TABLE:name}} replaced by the output of `new_directions/gp_tables.py` run on the stored cells
(nothing typed by hand).  python3 gp_report.py      (run from this directory; tables whose result directory does not exist are marked 'not run')"""
import os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ND = HERE.parents[1] / "new_directions"; S1, SB = str(HERE / "GP1_spec.json"), str(HERE / "GP1_blocked_spec.json")
NOISE = re.compile(r"warn|Validated|Loaded SimCLR|self\.ahat|xa, xb|Excluding", re.I)
TABLES = {
    "dev1": ["dev", "results/gp/dev1", "--only=gp_fixed,gp_ev,gp_evc,gp_cv"], "dev2": ["dev", "results/gp/dev2", "--only=gp_fix2,gp_cvt,gp_evt,gp_evct", "--consts_from=results/gp/dev3"],
    "grid_a": ["grid", "results/gp/grid1", "results/gp/grid2_aw8", "results/gp/grid2_aw16", "results/gp/grid2_aw32", "results/gp/grid3", "results/gp/grid7_aw16", "results/gp/grid7_aw32", "--top=12"],
    "grid_kernels": ["grid", "results/gp/grid4_arccos", "results/gp/grid4_mix", "results/gp/grid6_gdiff", "results/gp/grid5_gamma", "--top=14"],
    "dev45": ["dev", "results/gp/dev4", "--consts_from=results/gp/dev4"], "dev5": ["dev", "results/gp/dev5", "--consts_from=results/gp/dev5"],
    "devblk": ["dev", "results/gp/devblk", "--only=gp_fix2,gp_fix2_noG,gp_cvt_p,aw8"], "devcold": ["dev", "results/gp/devcold", "--only=gp_fix2,gp_fix2_noG,gp_cvt_p,aw8"],
    "variance_dev": ["variance", "results/gp/dev2", "--learner=gp_fix2"], "timing": ["timing", "results/gp/dev2"],
    "conf1": ["confirm", S1, "results/gp/conf1"], "conf2": ["confirm", S1, "results/gp/conf2"], "conf12": ["confirm", S1, "results/gp/conf1", "results/gp/conf2"],
    "blk1": ["confirm", SB, "results/gp/conf1_blk"], "blk2": ["confirm", SB, "results/gp/conf2_blk"], "cold1": ["confirm", S1, "results/gp/conf1_cold"],
    "bytype1": ["bytype", S1, "results/gp/conf1"], "bytype2": ["bytype", S1, "results/gp/conf2"], "bytype_dev": ["bytype", S1, "results/gp/dev4"], "strata_dev": ["strata", S1, "results/gp/dev4", "results/gp/dev5"], "strata1": ["strata", S1, "results/gp/conf1"], "strata2": ["strata", S1, "results/gp/conf2"], "variance_conf": ["variance", "results/gp/conf1", "results/gp/conf2", "--learner=gp_fix3"], "timing_conf": ["timing", "results/gp/conf1"],
}


def run(args) -> str:
    dirs = [a for a in args[1:] if a.startswith("results/")]
    if dirs and not all((ND / d).is_dir() and any((ND / d).glob("seed*.pkl.gz")) for d in dirs): return "*not run*"
    out = subprocess.run([sys.executable, "gp_tables.py", *args], cwd=ND, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{ND}:{ND.parents[1] / 'active_learning_program'}"})
    if out.returncode: return f"*failed*: {out.stderr[-400:]}"
    return "\n".join(l for l in out.stdout.splitlines() if not NOISE.search(l))


def main():
    text = (HERE / "results_narrative_graph_gp.md").read_text()
    def sub(m):
        name = m.group(1)
        return run(TABLES[name]) if name in TABLES else f"*unknown table {name}*"
    (HERE / "RESULTS_graph_gp.md").write_text(re.sub(r"\{\{TABLE:(\w+)\}\}", sub, text)); print("wrote RESULTS_graph_gp.md")


if __name__ == "__main__": main()

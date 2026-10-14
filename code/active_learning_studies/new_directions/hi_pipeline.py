"""One command for the whole honest-inference analysis (all steps frozen by PREREGISTRATION_HI1.md); writes results/hi/analysis/*.csv.  numpy / pandas / scipy / statsmodels only.
  python3 hi_pipeline.py [--B 10000] [--only main,strata,blocked,extra]  [--blocks conf=results/hi/conf1,results/hi/conf2 ...]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import hi_analyze as A
import hi_extra as X
import hi_infer as H

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
BLOCKS = {"conf": ["results/hi/conf1", "results/hi/conf2"], "conf1": ["results/hi/conf1"], "conf2": ["results/hi/conf2"], "dev": ["results/hi/dev"]}
BLOCKED = {"blocked_conf": ["results/hi/conf_blocked"], "blocked_dev": ["results/hi/dev_blocked"]}
KEY = ["C3", "C4", "C5"]


def exists(dirs): return all((HERE / d / "pred").is_dir() and any((HERE / d / "pred").glob("seed*.csv.gz")) for d in dirs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=10000); ap.add_argument("--B_alt", type=int, default=4000); ap.add_argument("--only", default="main,strata,blocked,extra"); ap.add_argument("--out", type=Path, default=HERE / "results" / "hi" / "analysis")
    ap.add_argument("--blocks", nargs="*", default=None); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); only = a.only.split(","); blocks = {k: v for k, v in BLOCKS.items() if exists(v)}
    if a.blocks: blocks = {s.split("=")[0]: s.split("=")[1].split(",") for s in a.blocks}
    comps = A.COMPS; t0 = time.monotonic()
    if "main" in only:
        for name, dirs in blocks.items():
            blk = A.load(dirs); units = ("unit_g", "unit_c0.03", "unit_b42") if name in ("conf", "dev") else ("unit_g",); df = A.analyze(blk, name, comps, ["A", "B"], ["auc", "acc", "ll"], ["all"], units, B=a.B, B_alt=a.B_alt, verbose=False)
            df.to_csv(a.out / f"main_{name}.csv", index=False); print(f"main {name}: {len(df)} rows, {time.monotonic() - t0:.0f}s", flush=True)
    if "strata" in only:
        for name in ("conf", "dev"):
            if name not in blocks: continue
            blk = A.load(blocks[name]); df = A.analyze(blk, name, comps, ["A", "B"], ["auc", "acc", "ll"], ["near", "mid", "far03", "far01", "htr"], ("unit_g",), B=a.B_alt, verbose=False); df.to_csv(a.out / f"strata_{name}.csv", index=False); print(f"strata {name}: {len(df)} rows, {time.monotonic() - t0:.0f}s", flush=True)
    if "blocked" in only:
        for name, dirs in BLOCKED.items():
            if not exists(dirs): continue
            blk = A.load(dirs); df = A.analyze(blk, name, comps, ["blocked"], ["auc", "acc", "ll"], ["all"], ("unit_g",), B=a.B_alt, verbose=False); df.to_csv(a.out / f"{name}.csv", index=False); print(f"{name}: {len(df)} rows, {time.monotonic() - t0:.0f}s", flush=True)
    if "extra" in only:
        for name in ("conf", "dev"):
            if name not in blocks: continue
            blk = A.load(blocks[name]); X.coverage(blk).to_csv(a.out / f"coverage_{name}.csv", index=False); ggs, rows, ses, vcs = [], [], [], []
            for cid in KEY + ["C2", "C9", "C11"]:
                _, cand, ref, _ = next(c for c in comps if c[0] == cid)
                for split in ("A", "B"):
                    gg = X.group_gains(blk, cand, ref, split); gg["comp"], gg["split"] = cid, split; ggs.append(gg.reset_index()); rows.append({"comp": cid, "split": split, **X.group_sign_summary(gg)})
                    for m in ("auc", "acc", "ll"): ses.append({"comp": cid, "split": split, "metric": m, **X.se_schemes(blk, cand, ref, split, m, B=2000)})
                    if cid in KEY:
                        for m in ("acc", "ll"):
                            try: vcs.append({"comp": cid, "split": split, "metric": m, **X.variance_components(blk, cand, ref, split, m)})
                            except Exception as e: print("variance components failed", cid, split, m, repr(e)[:100])
            pd.concat(ggs).to_csv(a.out / f"group_gains_{name}.csv", index=False); pd.DataFrame(rows).to_csv(a.out / f"group_signs_{name}.csv", index=False); pd.DataFrame(ses).to_csv(a.out / f"se_schemes_{name}.csv", index=False); pd.DataFrame(vcs).to_csv(a.out / f"variance_components_{name}.csv", index=False); print(f"extra {name}: {time.monotonic() - t0:.0f}s", flush=True)
    (a.out / "pipeline_info.json").write_text(json.dumps({"blocks": blocks, "B": a.B, "B_alt": a.B_alt, "only": only}, indent=1))


if __name__ == "__main__": main()

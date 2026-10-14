"""Build the pixel-level extraction cache: results/pf/extract.pt = {path: extract(path)} for every cached image.  python3 pf_build.py [--workers 4]"""
import sys, torch, multiprocessing as mp
from pathlib import Path
import nd_core as core, pf_extract as P

OUT = core.HERE / "results" / "pf"


VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"


def work(p): return p, P.extract(p, 0.70 if VERSION == "v1" else None)


if __name__ == "__main__":
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); paths = sorted(core.feature_cache())
    with mp.Pool(4) as pool: res = dict(pool.imap(work, paths, chunksize=8))
    torch.save(res, OUT / ("extract.pt" if VERSION == "v1" else f"extract_{VERSION}.pt")); print("saved", len(res))

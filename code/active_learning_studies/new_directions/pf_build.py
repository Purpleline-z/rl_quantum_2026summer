"""Build the pixel-level extraction cache: results/pf/extract.pt = {path: extract(path)} for every cached image.  python3 pf_build.py [--workers 4]"""
import sys, torch, multiprocessing as mp
from pathlib import Path
import nd_core as core, pf_extract as P

OUT = core.HERE / "results" / "pf"


def work(p): return p, P.extract(p)


if __name__ == "__main__":
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); paths = sorted(core.feature_cache())
    with mp.Pool(4) as pool: res = dict(pool.imap(work, paths, chunksize=8))
    torch.save(res, OUT / "extract.pt"); print("saved", len(res))

"""Descriptor blocks from the pixel-extraction cache:  python3 pf_blocks.py v1|v2  ->  results/pf/blocks_<v>.pt  (v1: horizon search window 5-70 %, v2: 5-40 %; see pf_extract.HORIZON_HI)"""
import sys, torch, numpy as np
import nd_core as core, pf_extract as P

if __name__ == "__main__":
    v = sys.argv[1]; ex = torch.load(core.HERE / "results" / "pf" / ("extract.pt" if v == "v1" else f"extract_{v}.pt"), weights_only=False); paths = sorted(ex); D = [P.descriptors(ex[p]) for p in paths]
    blocks = {k: np.stack([d[k] for d in D]) for k in D[0]}; torch.save({"paths": paths, "blocks": blocks}, core.HERE / "results" / "pf" / f"blocks_{v}.pt"); print(v, {k: v_.shape for k, v_ in blocks.items()}, "nan:", sum(int(np.isnan(x).sum()) for x in blocks.values()))

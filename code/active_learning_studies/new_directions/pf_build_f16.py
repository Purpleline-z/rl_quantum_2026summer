"""SimCLR features of the 18 ideal images that are stored as 16-bit PNGs, decoded correctly (value / 257 -> 8-bit scale, resampled to the 656x492 frame; ``pf_extract.load_gray``).  The repository's
loaders use ``Image.open(p).convert("L")``, which clips 16-bit values at 255, i.e. turns these images into saturated white discs (all 18 get the same feature, cosine 0.9999).
Output: results/pf/f16.pt = {path: feature} for those 18 images only.   python3 pf_build_f16.py"""
import tempfile, numpy as np, torch
from PIL import Image
import nd_core as core, pf_extract as P

if __name__ == "__main__":
    cache = core.feature_cache(); torch.set_num_threads(4); paths = [p for p in sorted(cache) if Image.open(p).mode == "I;16"]
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(10600, "A", scratch, cache); enc = ctx.features.base_model.encoder.eval(); tf = ctx.features.tf; out = {}
        with torch.no_grad():
            for p in paths:
                g, _ = P.load_gray(p); im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), mode="L"); out[p] = enc(tf(im)[None])[0].clone()
        # sanity: an 8-bit image run through the same path reproduces its cached feature
        q = next(p for p in sorted(cache) if "/Trajectories/" in p); g, _ = P.load_gray(q); im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), mode="L")
        with torch.no_grad(): chk = enc(tf(im)[None])[0]
        print("max |diff| 8-bit reproduction:", float((chk - cache[q].float()).abs().max()))
    (core.HERE / "results" / "pf").mkdir(parents=True, exist_ok=True); torch.save(out, core.HERE / "results" / "pf" / "f16.pt"); print("saved", len(out))

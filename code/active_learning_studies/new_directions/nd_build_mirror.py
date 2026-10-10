"""Encode the mirrored version of every cached image once (serial, so parallel runs never race on the file)."""
import tempfile
import nd_core as core, nd_learners as L
cache = core.feature_cache()
with tempfile.TemporaryDirectory() as s:
    ctx = core.make_ctx(2000, "A", s, cache)
    paths = sorted(cache); L.mirror_features(ctx, paths); print("mirrored", len(paths))

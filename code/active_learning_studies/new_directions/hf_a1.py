"""rt_a1.py with the Stage-2 learners of ``hf_learners.py`` registered (same CLI and output format):
  python3 hf_a1.py --seeds 10400-10419 --splits A,B --out results/hf/stage2_dev --learners lap3_k10_st4,hfs,hfq1 [--shard i/n]
Add ``--strict DELTA`` to evaluate on the anchor-strict context of rt_a9 (ideal images within cosine distance DELTA of a held-out image removed from every anchor table, for all learners alike).
"""
import sys
import hf_learners  # noqa: F401  (registers the learners in rt_core.REGISTRY)
import rt_a1


def main():
    if "--strict" in sys.argv:
        i = sys.argv.index("--strict"); delta = float(sys.argv[i + 1]); del sys.argv[i:i + 2]; install_strict(delta)
    rt_a1.main()


def install_strict(delta: float) -> None:
    """Wrap ``nd_core.make_ctx`` so every context has its anchor tables stripped of near-copies of held-out images (as in rt_a9.py)."""
    import nd_core as core, rt_core as rt
    sim = None; make = core.make_ctx

    def strict_ctx(seed, split, scratch, cache):
        nonlocal sim
        ctx = make(seed, split, scratch, cache); sim = sim or rt.Sim(cache); held = sorted(rt.heldout_images(ctx))
        for tag in ("references", "utility_images", "test_images"):
            table = getattr(ctx.exp, tag)
            for cls in list(table):
                ps = list(table[cls]); dist = 1 - sim.max_sim([str(p) for p in ps], held); table[cls] = [p for p, dm in zip(ps, dist) if dm > delta]
        return ctx
    core.make_ctx = strict_ctx


if __name__ == "__main__": main()

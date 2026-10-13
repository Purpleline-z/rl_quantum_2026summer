#!/usr/bin/env python3
"""nd_run.py with the representation learners (`rp_*`) added to the registry (same CLI):  python3 nd_run_rp.py --learners lap3_k10_st4,rp_s3 --seeds 8600-8609 --splits A,B --out results/rp/devA --shard 0/4"""
import nd_run
import nd_representation

_orig = nd_run.registry
nd_run.registry = lambda: {**_orig(), **nd_representation.LEARNERS}

if __name__ == "__main__":
    nd_run.main()

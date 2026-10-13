#!/usr/bin/env python3
"""nd_run.py with the graph-mechanism learners added to the registry (same CLI):  python3 nd_run_gm.py --learners aw8_all,lap3_k10,gm_lap3_n3 --seeds 8000-8019 --splits A,B --out results/gm/dev1"""
import nd_run
import nd_graph_mechanism

_orig = nd_run.registry
nd_run.registry = lambda: {**_orig(), **nd_graph_mechanism.LEARNERS}

if __name__ == "__main__":
    nd_run.main()

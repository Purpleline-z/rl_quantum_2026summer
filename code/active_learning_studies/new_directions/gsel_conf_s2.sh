#!/bin/bash
# Replication blocks of PREREGISTRATION_S2.md (fresh seeds 8340-8399). Idempotent; each block runs ONCE.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
ARMS=random,tdc,tdc_typeonly,tdc_shuffled,qcov_alloc,qcov_alloc_typeonly,qcov_alloc_shuffled
[ -f logs/gs_s2c.done ] || { for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $ARMS --learners aw8_all --seeds 8340-8369 --splits A,B --draws 5 --out results/gs_s2_cold --shard $i/4 >> logs/gs_s2c_$i.log 2>&1 & done; wait; echo done > logs/gs_s2c.done; }
[ -f logs/gs_s2r.done ] || { for i in 0 1 2 3; do python3 gsel_select.py --selectors $ARMS --learners aw8_all --seeds 8370-8399 --splits A,B --draws 5 --out results/gs_s2_regular --shard $i/4 >> logs/gs_s2r_$i.log 2>&1 & done; wait; echo done > logs/gs_s2r.done; }

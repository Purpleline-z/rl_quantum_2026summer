#!/bin/bash
# Larger replication blocks of PREREGISTRATION_S3.md (fresh seeds 8400-8479). Idempotent; each block runs ONCE.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
ARMS=random,tdc,tdc_typeonly,tdc_shuffled,qcov_alloc,qcov_alloc_shuffled
[ -f logs/gs_s3r.done ] || { for i in 0 1 2 3; do python3 gsel_select.py --selectors $ARMS --learners aw8_all --seeds 8400-8439 --splits A,B --draws 5 --out results/gs_s3_regular --shard $i/4 >> logs/gs_s3r_$i.log 2>&1 & done; wait; echo done > logs/gs_s3r.done; }
[ -f logs/gs_s3c.done ] || { for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $ARMS --learners aw8_all --seeds 8440-8479 --splits A,B --draws 5 --out results/gs_s3_cold --shard $i/4 >> logs/gs_s3c_$i.log 2>&1 & done; wait; echo done > logs/gs_s3c.done; }

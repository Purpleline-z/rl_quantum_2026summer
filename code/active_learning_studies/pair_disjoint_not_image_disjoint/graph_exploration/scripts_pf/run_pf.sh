#!/bin/bash
# usage: run_pf.sh OUTNAME LEARNERS SEEDS SPLITS [nshards] [ctx]   (idempotent: finished (seed, split, learner) cells are skipped; restart after a container restart)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/pf/$1; N=${5:-4}; CTX=${6:-standard}; mkdir -p $OUT logs_pf
for i in $(seq 0 $((N-1))); do nohup python3 pf_run.py --learners $2 --seeds $3 --splits $4 --ctx $CTX --out $OUT --shard $i/$N >> logs_pf/$1_$i.log 2>&1 & done

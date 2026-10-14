#!/bin/bash
# usage: run_cr.sh OUTNAME LEARNERS SEEDS KINDS BUDGETS [nshards]   (idempotent; restart after a container restart)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/cr/$1; N=${6:-4}; mkdir -p $OUT logs_cr
for i in $(seq 0 $((N-1))); do nohup python3 cr_run.py --learners $2 --seeds $3 --kinds $4 --budgets $5 --out $OUT --shard $i/$N >> logs_cr/$1_$i.log 2>&1 & done

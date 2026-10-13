#!/bin/bash
# usage: run_rg.sh OUTNAME LEARNERS SEEDS SPLITS DRAWS [nshards]   (idempotent; restart after a container restart)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/rg/$1; N=${6:-4}; mkdir -p $OUT logs_rg
for i in $(seq 0 $((N-1))); do nohup python3 rg_run.py --learners $2 --seeds $3 --splits $4 --draws $5 --out $OUT --shard $i/$N >> logs_rg/$1_$i.log 2>&1 & done

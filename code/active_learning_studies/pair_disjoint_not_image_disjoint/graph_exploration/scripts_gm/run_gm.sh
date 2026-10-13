#!/bin/bash
# usage: run_gm.sh OUTNAME LEARNERS SEEDS SPLITS [nshards]   (idempotent: finished cells are skipped; restart after a container restart)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/gm/$1; N=${5:-4}; mkdir -p $OUT logs_gm
for i in $(seq 0 $((N-1))); do nohup python3 nd_run_gm.py --learners $2 --seeds $3 --splits $4 --draws 3 --out $OUT --shard $i/$N >> logs_gm/$1_$i.log 2>&1 & done

#!/bin/bash
# usage: run_sel.sh OUTNAME SELECTORS LEARNERS SEEDS SPLITS [nshards]   (idempotent; restart after a container restart)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/sh/$1; N=${6:-4}; mkdir -p $OUT logs_sh
for i in $(seq 0 $((N-1))); do nohup python3 nd_select.py --selectors $2 --learners $3 --seeds $4 --splits $5 --draws 3 --out $OUT --shard $i/$N >> logs_sh/$1_$i.log 2>&1 & done

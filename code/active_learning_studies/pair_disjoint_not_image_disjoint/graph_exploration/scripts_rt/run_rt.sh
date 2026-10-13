#!/bin/bash
# usage: run_rt.sh SCRIPT OUTNAME [args...]   (idempotent; 4 shards; e.g. run_rt.sh rt_a1.py a1 --seeds 8800-8819 --splits A,B)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
S=$1; OUT=results/rt/$2; shift 2; mkdir -p logs_rt
for i in $(seq 0 3); do nohup python3 $S --out $OUT --shard $i/4 "$@" >> logs_rt/$(basename $OUT)_$i.log 2>&1 & done

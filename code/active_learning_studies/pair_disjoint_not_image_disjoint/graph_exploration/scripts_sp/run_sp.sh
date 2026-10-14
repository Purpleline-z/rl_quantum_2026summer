#!/bin/bash
# usage: run_sp.sh OUTNAME VARIANTS SEEDS DRAWS [SPLITS] [ND_INITIAL]   e.g. run_sp.sh dev1 lap3_k10_st4,fs_lam30 11200-11209 2 A,B groups
# 4 shards, idempotent (finished (budget, draw, variant) cells are skipped); logs in new_directions/logs_sp/
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=.:../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ND_INITIAL=${6:-groups}
OUT=results/sp/$1; mkdir -p logs_sp
for i in $(seq 0 3); do nohup python3 sp_run.py --variants "$2" --seeds "$3" --draws "$4" --splits "${5:-A,B}" --out $OUT --shard $i/4 >> logs_sp/$1_$i.log 2>&1 & done

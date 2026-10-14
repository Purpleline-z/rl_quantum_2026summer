#!/bin/bash
# usage: run_sa.sh OUTNAME LEARNERS SEEDS SPLITS DRAWS   (idempotent; 4 shards; one torch thread each)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /home/user/mm/env.sh 2>/dev/null
OUT=results/syn/$1; mkdir -p logs_sa
for i in $(seq 0 3); do nohup python3 syn_run.py --out $OUT --learners $2 --seeds $3 --splits $4 --draws $5 --shard $i/4 >> logs_sa/$1_$i.log 2>&1 & done

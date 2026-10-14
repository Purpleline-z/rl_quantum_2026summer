#!/bin/bash
# usage: run_td.sh MODE OUTNAME LEARNERS SEEDS [SPLITS]   (idempotent; 4 shards, one torch thread each)
# e.g.  run_td.sh forward dev1 BASE,L:trans_fut 10500-10519      run_td.sh standard devs1 BASE,L:trans_fut 10500-10519 A,B
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
source /home/user/mm/env.sh 2>/dev/null
OUT=results/td/$2; mkdir -p logs_td
for i in 0 1 2 3; do nohup python3 td_run.py --mode $1 --out $OUT --learners $3 --seeds $4 --splits ${5:-A,B} --shard $i/4 >> logs_td/$2_$i.log 2>&1 & done

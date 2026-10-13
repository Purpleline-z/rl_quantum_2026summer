#!/bin/bash
# usage: run_rc.sh OUTNAME LEARNERS SEEDS DRAWS   e.g.  run_rc.sh dev1 lap3_k10_st4,rc_fullg 9300-9309 2
# 4 shards, idempotent (finished (budget, draw, learner) cells are skipped); logs in new_directions/logs_rc/
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/rc/$1; mkdir -p logs_rc
for i in $(seq 0 3); do nohup python3 rc_run.py --learners "$2" --seeds "$3" --draws "$4" --out $OUT --shard $i/4 >> logs_rc/$1_$i.log 2>&1 & done

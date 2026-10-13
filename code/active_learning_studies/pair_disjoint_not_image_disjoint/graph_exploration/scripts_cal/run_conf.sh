#!/bin/bash
# usage: run_conf.sh   (idempotent; cells already written are skipped; restart after a container restart)
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p logs_cal
for i in 0 1 2 3; do python3 cal_run.py --seeds 9150-9184 --splits A,B --out results/cal/conf1 --draws 2 --folds 5 --shard $i/4 >> logs_cal/conf1_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do python3 cal_run.py --seeds 9150-9184 --splits A,B --out results/cal/conf1_aw8 --learner aw8_all_st2lr0.5 --draws 1 --folds 5 --shard $i/4 >> logs_cal/conf1_aw8_$i.log 2>&1 & done; wait
echo done > logs_cal/conf1.done

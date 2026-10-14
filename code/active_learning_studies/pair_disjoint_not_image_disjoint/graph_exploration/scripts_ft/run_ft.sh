#!/bin/bash
# usage: run_ft.sh OUTNAME LEARNERS SEEDS SPLITS BUDGETS DRAWS [NSHARDS]   e.g.  run_ft.sh dev1 lap3_k10_st4,ftp_base 10300-10307 A,B 20,40 1 4
# idempotent (finished (budget, draw, learner) cells are skipped); results in new_directions/results/ft/OUTNAME, logs in new_directions/logs_ft/
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/ft/$1; mkdir -p logs_ft; N=${7:-4}
for i in $(seq 0 $((N-1))); do nohup python3 ft_run.py --learners "$2" --seeds "$3" --splits "$4" --budgets "$5" --draws "$6" --out $OUT --shard $i/$N >> logs_ft/$1_$i.log 2>&1 & done

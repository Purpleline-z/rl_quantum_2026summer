#!/bin/bash
# EXPLORATORY (dev seeds 8700-8719): class balance of the anchors (stratified random diagnostic, balanced k-center) at 36 anchors; waits for the confirmation dose run
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ae
while [ ! -f logs/ae_conf_dose.done ]; do sleep 30; done
for i in 0 1 2 3; do python3 ae_run.py --mode active --seeds 8700-8719 --splits A,B --out results/ae/dev_active2 --rules rand0,rand1,rand2,strat,kcf_bal,kcf --shard $i/4 >> logs/ae_dev_active2_$i.log 2>&1 & done; wait
echo done > logs/ae_dev_active2.done

#!/bin/bash
# EXPLORATORY (dev seeds): is the harsh-split gain graph-specific or a generic score-shrinkage effect?  ridge1/ridge3 = no-graph shrinkage of random nodes' scores
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds run,temp,late --budgets 20,60 --draws 2 --arms base,lap3,ridge1,ridge3,lap3_shuf --out results/ge/dev_ridge --shard $i/4 >> logs/ge_dev_ridge_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_ridge.done

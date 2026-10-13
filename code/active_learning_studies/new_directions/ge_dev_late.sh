#!/bin/bash
# EXPLORATORY (dev seeds): forward-in-time harsh split (last third of every run held out)
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds late --budgets 20,60 --draws 2 --arms base,lap3,lap3_shuf,lap3_n3 --save-d --out results/ge/dev_late --shard $i/4 >> logs/ge_dev_late_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_late.done

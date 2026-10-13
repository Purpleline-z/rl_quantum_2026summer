#!/bin/bash
# EXPLORATORY (dev seeds): graph degree k = 5 / 20 on the harsh splits (lap3 with k 10 is the frozen candidate)
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds run,temp --budgets 20,60 --draws 2 --arms base,lap3,lap3_k5,lap3_k20 --out results/ge/dev_k --shard $i/4 >> logs/ge_dev_k_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_k.done

#!/bin/bash
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1; do nice -n 19 python3 ge_run_nodes.py --seeds 8400-8419 --splits A,B --budgets 20,60 --draws 3 --out results/ge/dev_nodes --shard $i/2 >> logs/ge_dev_nodes_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_nodes.done

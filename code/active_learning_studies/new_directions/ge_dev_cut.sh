#!/bin/bash
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1 2 3; do python3 ge_run_cut.py --seeds 8400-8419 --out results/ge/dev_cut --shard $i/4 >> logs/ge_dev_cut_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_cut.done

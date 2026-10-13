#!/bin/bash
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1; do nice -n 19 python3 ge_run_lamclass.py --seeds 8400-8419 --out results/ge/dev_lamclass --shard $i/2 >> logs/ge_dev_lamclass_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_lamclass.done

#!/bin/bash
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
while [ ! -f logs/ge_dev_rest.done ]; do sleep 20; done
for i in 0 1 2 3; do python3 ge_run_lam.py --seeds 8400-8419 --splits A,B --budgets 20,60 --draws 2 --out results/ge/dev_lam --shard $i/4 >> logs/ge_dev_lam_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_lam.done

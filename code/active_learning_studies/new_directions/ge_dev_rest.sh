#!/bin/bash
# dev screens for dose-response and harsh splits (idempotent); waits for the classification dev screen
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
while [ ! -f logs/ge_dev_class.done ]; do sleep 20; done
for i in 0 1 2 3; do python3 ge_run_dose.py --seeds 8400-8419 --splits A,B --budgets 20,60 --draws 3 --out results/ge/dev_dose --shard $i/4 >> logs/ge_dev_dose_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds run,temp --budgets 20,60 --draws 2 --out results/ge/dev_harsh --shard $i/4 >> logs/ge_dev_harsh_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_rest.done

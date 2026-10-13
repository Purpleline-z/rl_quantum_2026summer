#!/bin/bash
# EXPLORATORY (dev seeds): three temperature terciles in turn, more test groups; waits for confirmation 2
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
while [ ! -f logs/ge_conf2_dose.done ]; do sleep 30; done
for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds temp3 --budgets 20,60 --draws 2 --arms base,lap3,lap3_shuf,lap3_n3 --save-d --out results/ge/dev_temp3 --shard $i/4 >> logs/ge_dev_temp3_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_temp3.done

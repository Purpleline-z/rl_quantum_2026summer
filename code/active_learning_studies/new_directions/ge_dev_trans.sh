#!/bin/bash
# EXPLORATORY (dev seeds): harsh splits with the held-out condition's UNLABELLED frames (and the test images) as graph nodes (lap3_T) vs the inductive graph (lap3); waits for temp3
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
while [ ! -f logs/ge_dev_temp3.done ]; do sleep 30; done
for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds run,temp --budgets 20,60 --draws 2 --arms base,lap3,lap3_T --out results/ge/dev_trans --shard $i/4 >> logs/ge_dev_trans_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_trans.done

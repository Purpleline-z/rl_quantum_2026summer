#!/bin/bash
# EXPLORATORY (dev seeds): dose-response of the held-out condition's UNLABELLED frames in the graph (0 / 25 / 50 / 100% of the frames, test images excluded; plus the transductive arm with the test images)
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge
for i in 0 1 2; do python3 ge_run_harsh.py --seeds 8400-8419 --kinds run,temp --budgets 20,60 --draws 2 --arms base,lap3,lapF25,lapF50,lapF100,lap3_T --out results/ge/dev_target --shard $i/3 >> logs/ge_dev_target_$i.log 2>&1 & done; wait
echo done > logs/ge_dev_target.done

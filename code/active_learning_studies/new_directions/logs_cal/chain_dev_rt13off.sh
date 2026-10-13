#!/bin/bash
cd /home/user/rl_quantum_2026summer/code/active_learning_studies/new_directions
export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while [ ! -f logs_cal/chain_dev_baselines.done ]; do sleep 15; done
L=lap3_k10_st4_rt13off
for i in 0 1 2 3; do python3 cal_run.py --seeds 9100-9119 --splits A,B --out results/cal/dev_$L --learner $L --draws 1 --folds 5 --shard $i/4 >> logs_cal/dev_$L.$i.log 2>&1 & done
wait; echo done > logs_cal/chain_dev_rt13off.done

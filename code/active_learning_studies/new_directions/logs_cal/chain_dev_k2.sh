#!/bin/bash
cd /home/user/rl_quantum_2026summer/code/active_learning_studies/new_directions
export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for K in 3 10; do
  for i in 0 1 2 3; do python3 cal_run.py --seeds 9100-9109 --splits A,B --out results/cal/dev_k$K --draws 1 --folds $K --shard $i/4 >> logs_cal/dev_k$K.$i.log 2>&1 & done
  wait
done
echo done > logs_cal/chain_dev_k.done

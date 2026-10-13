#!/bin/bash
# development screens (idempotent; skips finished seed x split files)
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ae
run() { for i in 0 1 2 3; do python3 ae_run.py --mode $1 --seeds $2 --splits A,B --out results/ae/$3 --shard $i/4 >> logs/ae_$3_$i.log 2>&1 & done; wait; echo done > logs/ae_$3.done; }
[ -f logs/ae_dev_active.done ] || run active 8700-8713 dev_active
[ -f logs/ae_dev_dose.done ] || run dose 8700-8709 dev_dose
[ -f logs/ae_dev_lam.done ] || run lam 8700-8709 dev_lam

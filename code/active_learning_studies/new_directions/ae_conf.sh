#!/bin/bash
# Confirmation 1 of PREREGISTRATION_E1 (idempotent; each block runs ONCE).  Waits for the development chain so that the cores are not oversubscribed.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ae
while [ ! -f logs/ae_dev_lam.done ]; do sleep 30; done
run() { for i in 0 1 2 3; do python3 ae_run.py --mode $1 --seeds 8750-8784 --splits A,B --out results/ae/$2 --rules $3 --shard $i/4 >> logs/ae_$2_$i.log 2>&1 & done; wait; echo done > logs/ae_$2.done; }
[ -f logs/ae_conf_active.done ] || run active conf_active rand0,rand1,rand2,htrq,htrq_shuf,kcf,kcg,kcg_shuf,typi
[ -f logs/ae_conf_dose.done ] || run dose conf_dose rand0

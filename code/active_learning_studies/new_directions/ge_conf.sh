#!/bin/bash
# Confirmation block of PREREGISTRATION_GE1 (idempotent; each block runs ONCE).  usage: ge_conf.sh NAME SEEDS     e.g.  ge_conf.sh conf1 8500-8534
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs results/ge; N=$1; S=$2
[ -f logs/ge_${N}_class.done ] || { for i in 0 1 2 3; do python3 ge_run_class.py --seeds $S --confirm --out results/ge/${N}_class --shard $i/4 >> logs/ge_${N}_class_$i.log 2>&1 & done; wait; echo done > logs/ge_${N}_class.done; }
[ -f logs/ge_${N}_harsh.done ] || { for i in 0 1 2 3; do python3 ge_run_harsh.py --seeds $S --kinds run,temp --budgets 10,20,40,60 --draws 2 --arms base,lap3,lap3_shuf,lap3_n3 --save-d --out results/ge/${N}_harsh --shard $i/4 >> logs/ge_${N}_harsh_$i.log 2>&1 & done; wait; echo done > logs/ge_${N}_harsh.done; }
[ -f logs/ge_${N}_dose.done ] || { for i in 0 1 2 3; do python3 ge_run_dose.py --seeds $S --splits A,B --budgets 10,20,40,60 --draws 2 --out results/ge/${N}_dose --shard $i/4 >> logs/ge_${N}_dose_$i.log 2>&1 & done; wait; echo done > logs/ge_${N}_dose.done; }

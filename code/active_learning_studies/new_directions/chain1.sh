#!/bin/bash
# waits for the running screen2b shards, then runs screen4 (ridge BT, MLP+GP ensemble) and sel3 (selection with the GP learner in the loop)
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1
while pgrep -f "nd_run.py --learners baseline,gp_preference" > /dev/null; do sleep 20; done
for i in 0 1 2 3; do python3 nd_run.py --learners baseline,ridge_bt,ensemble_mlp_gp --seeds 2000-2009 --splits A,B --draws 3 --out results/screen4 --shard $i/4 > logs/screen4_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do python3 nd_select.py --selectors random,vopt_u,vopt_raw --learners baseline,gp_preference --seeds 2000-2009 --splits A,B --draws 3 --out results/sel3 --shard $i/4 > logs/sel3_$i.log 2>&1 & done; wait
echo chain1 done > logs/chain1.done

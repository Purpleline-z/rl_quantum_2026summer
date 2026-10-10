#!/bin/bash
# after the confirmatory run: dev screens of mixture anchors, combinations with aw8_all, learner-consistent selection, cold start
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1
while pgrep -f "out results/confirm1" > /dev/null; do sleep 20; done
for i in 0 1 2 3; do python3 nd_run.py --learners baseline,aw8_all,aw8_all_mix1,aw8_all_mix4,aw025_all_mix4 --seeds 2000-2019 --splits A,B --draws 3 --out results/screen11_mix --shard $i/4 > logs/screen11_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do python3 nd_select.py --selectors random,vopt_u,vopt_aw8_all --learners baseline,aw8_all --seeds 2000-2019 --splits A,B --draws 3 --out results/sel6 --shard $i/4 > logs/sel6_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do ND_INITIAL=random python3 nd_select.py --selectors random,vopt_u --learners baseline,aw8,aw8_all --seeds 2000-2019 --splits A,B --draws 3 --out results/sel5_cold --shard $i/4 > logs/sel5_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do python3 nd_run.py --learners baseline,aw8_all,sym_aw8_all,ens5_aw8_all,meta_aw8_all,gp_aw8 --seeds 2000-2009 --splits A,B --draws 3 --out results/screen10 --shard $i/4 > logs/screen10_$i.log 2>&1 & done; wait
echo chain4 done > logs/chain4.done

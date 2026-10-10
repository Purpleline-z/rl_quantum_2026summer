#!/bin/bash
# waits for chain1, then runs sel4 (committee of MLP head and GP) on the dev seeds
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1
while [ ! -f logs/chain1.done ]; do sleep 30; done
for i in 0 1 2 3; do python3 nd_select.py --selectors random,vopt_u,qbc_gp,vopt_qbc --learners baseline --seeds 2000-2019 --splits A,B --draws 3 --out results/sel4 --shard $i/4 > logs/sel4_$i.log 2>&1 & done; wait
echo chain2 done > logs/chain2.done

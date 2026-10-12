#!/bin/bash
# cold start (10 random initial judgments instead of the 10 initial groups): selectors and learners on 10 dev seeds
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 ND_INITIAL=random
while [ ! -f logs/chain2.done ]; do sleep 30; done
for i in 0 1 2 3; do python3 nd_select.py --selectors random,vopt_u,vopt_raw,vopt_stoch3,qbc_gp,vopt_qbc --learners baseline,gp_preference,ensemble5 --seeds 2000-2009 --splits A,B --draws 3 --out results/sel5_cold --shard $i/4 > logs/sel5_$i.log 2>&1 & done; wait
echo chain3 done > logs/chain3.done

#!/bin/bash
# Dev screen 3 (idempotent): non-graph prior-strength controls and decisive-weighted design, regular + cold start. Waits for dev2.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_dev2.done ]; do sleep 20; done
SEL=random,vopt_ridge4@aw,vopt_ridge10@aw,vopt_dec@aw,vopt_dec_typeonly@aw,vopt_dec_shuffled@aw
for i in 0 1 2 3; do python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev3 --shard $i/4 >> logs/gs_dev3_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev3_cold --shard $i/4 >> logs/gs_dev3_cold_$i.log 2>&1 & done; wait
echo done > logs/gs_dev3.done

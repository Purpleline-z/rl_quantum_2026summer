#!/bin/bash
# Dev screen 1 (idempotent): frozen graph rules + controls under the aw8_all learner, dev seeds 8000-8019, regular and cold start.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
SEL=random,vopt_u,vopt_u@aw,tdc,tdc_typeonly,tdc_shuffled,tcu,gvopt_lap3,gvopt_lap3@aw,gvopt_lap3_shuffled@aw
for i in 0 1 2 3; do python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev1 --shard $i/4 >> logs/gs_dev1_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev1_cold --shard $i/4 >> logs/gs_dev1_cold_$i.log 2>&1 & done; wait
echo done > logs/gs_dev1.done

#!/bin/bash
# Dev screen 2 (idempotent): HTR-oriented graph rules + class-priority (no graph) control + shuffled-graph controls, aw8_all learner, regular start. Waits for dev1.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_dev1.done ]; do sleep 20; done
SEL=random,vopt_u@aw,htr_vopt_prio@aw,htr_amb,htr_amb_shuffled,htr_bridge,htr_bridge_shuffled,htr_cov_amb,htr_cov_amb_shuffled,htr_cov_bridge,htr_cov_bridge_shuffled,htr_vopt_amb@aw,htr_vopt_amb_shuffled@aw,htr_vopt_bridge@aw,htr_vopt_bridge_shuffled@aw
for i in 0 1 2 3; do python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev2 --shard $i/4 >> logs/gs_dev2_$i.log 2>&1 & done; wait
echo done > logs/gs_dev2.done

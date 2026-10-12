#!/bin/bash
# Dev screen 4 (idempotent): cold start for the HTR-oriented coverage rules and the class-priority control. Waits for dev3.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_dev3.done ]; do sleep 20; done
SEL=random,htr_cov_amb,htr_cov_amb_shuffled,htr_cov_bridge,htr_cov_bridge_shuffled,htr_vopt_prio@aw
for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev4_cold --shard $i/4 >> logs/gs_dev4_$i.log 2>&1 & done; wait
echo done > logs/gs_dev4.done

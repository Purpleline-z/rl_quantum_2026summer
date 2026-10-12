#!/bin/bash
# Confirmation blocks of PREREGISTRATION_S1.md (idempotent: finished cells are skipped). Waits for dev5. Each block runs ONCE.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_dev5.done ]; do sleep 20; done
A12=random,vopt_u@aw,gvopt_lap3@aw,tdc,tdc_shuffled,tdc_typeonly,gvopt_lap3_shuffled@aw,vopt_ridge10@aw
B3=random,vopt_u@aw,gvopt_lap3@aw,tdc,htr_cov_amb,htr_cov_amb_shuffled,htr_vopt_prio@aw,vopt_ridge10@aw
for i in 0 1 2 3; do python3 gsel_select.py --selectors $A12 --learners aw8_all --seeds 8100-8134 --splits A,B --draws 5 --out results/gs_conf1 --shard $i/4 >> logs/gs_conf1_$i.log 2>&1 & done; wait; echo done > logs/gs_conf1.done
for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $A12 --learners aw8_all --seeds 8200-8234 --splits A,B --draws 5 --out results/gs_conf2_cold --shard $i/4 >> logs/gs_conf2_$i.log 2>&1 & done; wait; echo done > logs/gs_conf2.done
for i in 0 1 2 3; do python3 gsel_select.py --selectors $B3 --learners aw8_all --seeds 8300-8334 --splits A,B --draws 5 --out results/gs_conf3_htr --shard $i/4 >> logs/gs_conf3_$i.log 2>&1 & done; wait; echo done > logs/gs_conf3.done

#!/bin/bash
# after chain_dev2.sh: schedule re-tuning (steps / learning rate) of the Laplacian learner and of aw8_all; idempotent
cd "$(dirname "$0")/.."
while [ ! -f ../../new_directions/results/sh/CHAIN_DEV2_DONE ]; do sleep 30; done
wait_idle() { while [ $(ps aux | grep "nd_select.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
bash scripts_sh/run_sel.sh sdev5 random,vopt_u lap3_k10_st3,lap3_k10_st4,lap3_k10_st2lr2,aw8_all_st2,aw8_all_st3,aw8_all_st4 8200-8219 A,B 4; sleep 30; wait_idle
echo done > ../../new_directions/results/sh/CHAIN_DEV3_DONE

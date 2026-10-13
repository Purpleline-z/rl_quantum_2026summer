#!/bin/bash
# development screens after sdev1: sdev2 (new selectors), sdev3 (learner changes); idempotent
cd "$(dirname "$0")/.."
wait_idle() { while [ $(ps aux | grep "nd_select.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
wait_idle
bash scripts_sh/run_sel.sh sdev2 random,vopt_u,gdens_unc,gdens_unc_shuf,gcover,gcover_dec,gcover_dec_shuf,gvopt_node1,gvopt_node3,gvopt_node3_shuf,gvopt_node10,vopt_htr2,vopt_htr4,vopt_bridge1,vopt_bridge3,vopt_bridge3_shuf lap3_k10,aw8_all 8200-8219 A,B 4; sleep 30; wait_idle
bash scripts_sh/run_sel.sh sdev3 random,vopt_u lap3h10_k10,lap3_k10_st0.5,lap3_k10_st2,lap3_k10_lr0.5,lap3_k10_lr2 8200-8219 A,B 4; sleep 30; wait_idle
echo done > ../../new_directions/results/sh/CHAIN_DEV_DONE

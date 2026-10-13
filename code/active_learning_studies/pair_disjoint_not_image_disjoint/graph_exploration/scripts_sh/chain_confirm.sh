#!/bin/bash
# confirmation runs in the order of PREREGISTRATION_SH1.md; idempotent (restart after a container restart)
cd "$(dirname "$0")/.."
wait_idle() { while [ $(ps aux | grep "nd_select.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
SEL=random,vopt_u,gvopt_node3,gvopt_node3_shuf,typed_decisive_bald,typed_decisive_bald_shuffled,vopt_htr4
LRN=aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4,lap3_k10_st4_shuf
wait_idle
bash scripts_sh/run_sel.sh c1_sel $SEL lap3_k10 8300-8334 A,B 4; sleep 30; wait_idle
bash scripts_sh/run_sel.sh c1_learn random $LRN 8300-8334 A,B 4; sleep 30; wait_idle
bash scripts_sh/run_sel.sh c2_sel $SEL lap3_k10 8350-8384 A,B 4; sleep 30; wait_idle
bash scripts_sh/run_sel.sh c2_learn random $LRN 8350-8384 A,B 4; sleep 30; wait_idle
bash scripts_sh/run_sel.sh c1_sel4 $SEL lap3_k10_st4 8300-8334 A,B 4; sleep 30; wait_idle
echo done > ../../new_directions/results/sh/CHAIN_CONFIRM_DONE

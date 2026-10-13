#!/bin/bash
# cold-start confirmation (PREREGISTRATION_SH2) after the SH1 chain; idempotent
cd "$(dirname "$0")/.."
while [ ! -f ../../new_directions/results/sh/CHAIN_CONFIRM_DONE ]; do sleep 30; done
wait_idle() { while [ $(ps aux | grep "nd_select.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
ND_INITIAL=random bash scripts_sh/run_sel.sh cold_learn random aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4,lap3_k10_st4_shuf 8385-8399 A,B 4; sleep 30; wait_idle
echo done > ../../new_directions/results/sh/CHAIN_COLD_DONE

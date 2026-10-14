#!/bin/bash
# confirmation runs of PREREGISTRATION_SA1 (one draw, run once each).  usage: chain_confirm.sh NAME SEEDS   e.g. chain_confirm.sh confirm1 11130-11164
cd "$(dirname "$0")"; source /home/user/mm/env.sh 2>/dev/null
L=lap3_k10_st4,sy_gate_mask,sy_gate_base2,sy_ens_mask,sy_ens_base2,sy_tm_a10,sy_tm_a10_perm,sy_tmg_a10
bash run_sa.sh $1 $L $2 A,B 1; sleep 30
while [ $(ps -eo args | grep "[s]yn_run.py --out results/syn/$1 " | wc -l) -gt 0 ]; do sleep 30; done
bash run_sa.sh $1_blocked lap3_k10_st4,sy_gate_mask,sy_ens_mask,sy_tm_a10 $2 blocked 1

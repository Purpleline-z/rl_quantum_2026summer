#!/bin/bash
# runs confirmation 1 then confirmation 2 (idempotent; restart after a container restart)
cd "$(dirname "$0")/.."
L=aw8_all,lap3_k10,gm_lap3_n0,gm_lap3_n0.5,gm_lap3_n1,gm_lap3_n2,gm_lap3_n3,gm_lap10_n0,gm_lap10_n0.5,gm_lap10_n1,gm_lap10_n2,gm_lap10_n3,gm_lap10_n0_shuf,gmc_t_no_rt13,gmc_t_no_rt13_shuf,gmc_t_no_rt13_n3,gm_noise10
wait_idle() { while [ $(ps aux | grep "nd_run_gm.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
wait_idle; bash scripts_gm/run_gm.sh confirm1 $L 8100-8134 A,B 4; sleep 30; wait_idle
bash scripts_gm/run_gm.sh confirm2 $L 8150-8184 A,B 4; sleep 30; wait_idle
echo chain done > ../../new_directions/results/gm/CHAIN_DONE

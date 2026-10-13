#!/bin/bash
# after chain_dev.sh: HTR anchor-weight learner variants and the cold-start development look; idempotent
cd "$(dirname "$0")/.."
while [ ! -f ../../new_directions/results/sh/CHAIN_DEV_DONE ]; do sleep 30; done
wait_idle() { while [ $(ps aux | grep "nd_select.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
bash scripts_sh/run_sel.sh sdev4 random,vopt_u lap3_ha2_k10,lap3_ha4_k10,lap3_ha2h10_k10 8200-8219 A,B 4; sleep 30; wait_idle
echo done > ../../new_directions/results/sh/CHAIN_DEV2_DONE

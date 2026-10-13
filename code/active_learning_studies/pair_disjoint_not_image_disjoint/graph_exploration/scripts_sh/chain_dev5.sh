#!/bin/bash
# after the cold-start confirmation: graph-free regulariser at the long schedule (development robustness check, seeds 8200-8219); idempotent
cd "$(dirname "$0")/.."
while [ ! -f ../../new_directions/results/sh/CHAIN_COLD_DONE ]; do sleep 30; done
bash scripts_sh/run_sel.sh sdev9 random cons0.3_st4,cons1_st4 8200-8219 A,B 4

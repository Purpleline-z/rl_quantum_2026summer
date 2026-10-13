#!/bin/bash
# after chain1: audit 2 (graph de-duplication); idempotent
cd "$(dirname "$0")/.."
while [ ! -f ../../new_directions/results/rt/CHAIN1_DONE ]; do sleep 30; done
wait_idle() { while [ $(ps aux | grep -E "rt_a[0-9]\.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
bash scripts_rt/run_rt.sh rt_a2.py a2 --seeds 8800-8819 --splits A,B; sleep 30; wait_idle
echo done > ../../new_directions/results/rt/CHAIN2_DONE

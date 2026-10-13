#!/bin/bash
# after audit 1: audit 3 (time-blocked evaluation); idempotent
cd "$(dirname "$0")/.."
wait_idle() { while [ $(ps aux | grep -E "rt_a[0-9]\.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
sleep 20; wait_idle
bash scripts_rt/run_rt.sh rt_a3.py a3 --seeds 8800-8819; sleep 30; wait_idle
echo done > ../../new_directions/results/rt/CHAIN1_DONE

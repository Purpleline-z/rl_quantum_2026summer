#!/bin/bash
# after chain2: audit 4 (independent re-implementation) and audit 5 (randomness decomposition); idempotent
cd "$(dirname "$0")/.."
while [ ! -f ../../new_directions/results/rt/CHAIN2_DONE ]; do sleep 30; done
wait_idle() { while [ $(ps aux | grep -E "rt_(a[0-9]|reimpl)\.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
bash scripts_rt/run_rt.sh rt_reimpl.py a4 --seeds 8800-8819 --splits A,B; sleep 30; wait_idle
bash scripts_rt/run_rt.sh rt_a5.py a5 --seeds 8800-8809 --splits A,B; sleep 30; wait_idle
echo done > ../../new_directions/results/rt/CHAIN3_DONE

#!/bin/bash
# red-team confirmation runs (PREREGISTRATION_RT1.md); idempotent
cd "$(dirname "$0")/.."
wait_idle() { while [ $(ps aux | grep -E "rt_(a[0-9]|reimpl)\.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
bash scripts_rt/run_rt.sh rt_a1.py c_a1 --seeds 8850-8884 --splits A,B --learners baseline,aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4; sleep 30
while [ $(ps aux | grep -E "rt_a1\.py.*c_a1" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done
(cd ../../new_directions && PYTHONPATH=../../active_learning_program python3 rt_a1b.py --seeds 8850-8884 --splits A,B --out results/rt/c_a1b)
bash scripts_rt/run_rt.sh rt_a3.py c_a3 --seeds 8850-8884 --learners aw8_all_st2lr0.5,lap3_k10_st4,lap3_k10_st4_shuf; sleep 30
while [ $(ps aux | grep -E "rt_a3\.py.*c_a3" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done
bash scripts_rt/run_rt.sh rt_a2.py c_a2 --seeds 8850-8884 --splits A,B --deltas 0,0.003; sleep 30
while [ $(ps aux | grep -E "rt_a2\.py.*c_a2" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done
echo done > ../../new_directions/results/rt/CHAIN_CONFIRM_DONE

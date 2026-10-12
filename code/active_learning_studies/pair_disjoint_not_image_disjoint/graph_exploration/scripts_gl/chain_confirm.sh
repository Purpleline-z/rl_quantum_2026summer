#!/bin/bash
# runs confirmation 1, confirmation 2, cold-start confirmation one after another (idempotent; restart after a container restart)
cd "$(dirname "$0")/.."; export PYTHONPATH=../../../active_learning_program
L=aw8_all,lap1_k10,lap3_k10,lap10_k10,lap3_rq3_k10,lap1_k10_shuf,lap3_k10_shuf,lap10_k10_shuf,lap3_rq3_k10_shuf,lap1_k10_n3,lap3_k10_n3,lap10_k10_n3
wait_idle() { while [ $(ps aux | grep "nd_run.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
wait_idle; bash scripts_gl/run_nd.sh confirm1 $L 7100-7134 A,B 4; sleep 30; wait_idle
bash scripts_gl/run_nd.sh confirm2 $L 7200-7234 A,B 4; sleep 30; wait_idle
ND_INITIAL=random bash scripts_gl/run_nd.sh cold $L 7300-7334 A,B 4; sleep 30; wait_idle
echo chain done > ../../new_directions/results/gl/CHAIN_DONE

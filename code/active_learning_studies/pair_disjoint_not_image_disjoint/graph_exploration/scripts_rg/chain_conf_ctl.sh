#!/bin/bash
# waits for the main confirmation shards (9681 9682 9683 9684 9725 9726 ), then runs the shuffled-graph controls (draw 0 only)
while [ -n "$(ps -p 9681,9682,9683,9684 -o pid= 2>/dev/null)" ]; do sleep 20; done
bash "$(dirname "$0")/run_rg.sh" conf_ctl lap3_k10_st4_shuf,rglap3_cons0.3_k10_shuf_st4 9050-9084 A,B 1 4

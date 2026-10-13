#!/bin/bash
# after sdev6: selection under the re-tuned Laplacian learner (sdev7) and cold-start look (sdev8); idempotent
cd "$(dirname "$0")/.."
wait_idle() { while [ $(ps aux | grep "nd_select.py" | grep -v grep | wc -l) -gt 0 ]; do sleep 20; done; }
sleep 20; wait_idle
bash scripts_sh/run_sel.sh sdev7 random,vopt_u,gvopt_node3,gvopt_node3_shuf,typed_decisive_bald,typed_decisive_coverage,vopt_htr4,vopt_bridge1,vopt_bridge1_shuf,gcover_dec,gcover_dec_shuf lap3_k10_st4 8200-8219 A,B 4; sleep 30; wait_idle
ND_INITIAL=random bash scripts_sh/run_sel.sh sdev8cold random,vopt_u,gvopt_node3,typed_decisive_bald,vopt_htr4 lap3_k10_st4,lap3_k10,aw8_all 8200-8219 A,B 4; sleep 30; wait_idle
echo done > ../../new_directions/results/sh/CHAIN_DEV4_DONE

#!/bin/bash
# waits for the stage-A workers to finish, then launches stage B1 (shuffled controls of the physics-graph variants, other physics feature sets for the graph metric, logistic-regression anchor posterior)
while pgrep -f "pf_run.py.*results/pf/[d]evA" > /dev/null; do sleep 20; done
bash "$(dirname "$0")/run_pf.sh" devB1 pf_pfa_g_shuf,pf_pfa_gs_shuf,pf_pfb_g,pf_pfc_g,pf_pfb_gs,pf_pfc_gs,pf_qlr_pfa,pf_qlr_s4,pf_q_pfa_shuf 10600-10607 A,B 4

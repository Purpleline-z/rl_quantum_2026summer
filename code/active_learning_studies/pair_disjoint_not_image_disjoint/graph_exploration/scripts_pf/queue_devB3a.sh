#!/bin/bash
# waits for stage B2, then launches stage B3a: Laplacian penalty off for the root-13 type (confirmed elsewhere for the SimCLR graph) on the SimCLR graph (control) and on the fused physics graphs
while pgrep -f "pf_run.py.*results/pf/[d]evB2" > /dev/null; do sleep 20; done
bash "$(dirname "$0")/run_pf.sh" devB3a pg_s4_k10_l3_nort13,pg_pfa2s_k10_l3_nort13,pg_pfc2s_k10_l3_nort13 10600-10607 A,B 4

#!/bin/bash
# waits for stage B1, then launches stage B2: features v2 (horizon fix) as graph metric, k / lambda variants, random-projection graph control, anchor-posterior variant
while pgrep -f "pf_run.py.*results/pf/[d]evB1" > /dev/null; do sleep 20; done
bash "$(dirname "$0")/run_pf.sh" devB2 pg_pfa2_k10_l3,pg_pfa2s_k10_l3,pg_pfc2s_k10_l3,pg_rpp_k10_l3,pg_pfa2s_k10_l1,pg_pfa2s_k10_l10,pg_pfa2s_k5_l3,pg_pfa2s_k20_l3,pf_q_pfa2 10600-10607 A,B 4

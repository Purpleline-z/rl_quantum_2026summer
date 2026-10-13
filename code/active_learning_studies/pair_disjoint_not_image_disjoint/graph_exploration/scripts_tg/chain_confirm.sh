#!/bin/bash
# confirmation runs of PREREGISTRATION_TG1 (seeds 9250-9284, one draw, run once); A/B first, then the red-team blocked split.  Idempotent.
cd "$(dirname "$0")"; source /home/user/mm/env.sh 2>/dev/null
L=lap3_k10_st4,tg_dec5,tg_dec5_shuf,tg_dec5_mu0.3,tg_dec5_mu0.3_shuf,tg_curv,tg_curv_shuf
bash run_tg.sh confirm $L 9250-9284 A,B 1; sleep 30
while [ $(ps aux | grep "[t]g_run.py --out results/tg/confirm " | wc -l) -gt 0 ]; do sleep 30; done
bash run_tg.sh confirm_blocked lap3_k10_st4,tg_dec5,tg_dec5_mu0.3,tg_curv 9250-9284 blocked 1

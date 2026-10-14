#!/bin/bash
# pre-registered confirmation runs of PREREGISTRATION_TD1 (see TD1_runs.txt); each block once, idempotent (finished cells are skipped).  Usage: chain_confirm.sh 1|2
cd "$(dirname "$0")"; source /home/user/mm/env.sh 2>/dev/null
waitfor() { sleep 40; while pgrep -f "^python3 td_run.py.*--out results/td/$1( |$)" > /dev/null; do sleep 30; done; }
if [ "$1" = 1 ]; then
  LF="BASE,STD,G0,C:trans_win:c0.3,AQ:ind:r0.3,L:trans_fut,L:trans_win,L:trans_test,AQ:trans_win:r0.3,AQ:trans_test:r0.3,L:trans_win:shuf,L:trans_test:shuf,AQ:trans_win:r0.3:shuf,AQ:trans_test:r0.3:shuf,L:trans_win:tshuf,L:trans_test:tshuf,AQ:trans_win:r0.3:tshuf,AQ:trans_test:r0.3:tshuf"
  LS="BASE,G0,C:trans_fut:c0.3,L:trans_fut,L:trans_fut:shuf,L:trans_fut:tshuf,AQ:trans_fut:r0.3,AQ:trans_fut:r0.3:shuf,AQ:trans_fut:r0.3:tshuf"
  bash run_td.sh forward confirm1_fwd $LF 10530-10564; waitfor confirm1_fwd
  bash run_td.sh standard confirm1_std $LS 10530-10564 A,B; waitfor confirm1_std
  bash run_td.sh blocked confirm1_blk $LS 10530-10564; waitfor confirm1_blk
else
  LF="BASE,G0,C:trans_win:c0.3,L:trans_win,L:trans_test,AQ:trans_win:r0.3,AQ:trans_test:r0.3,L:trans_win:shuf,L:trans_test:shuf,AQ:trans_win:r0.3:shuf,AQ:trans_test:r0.3:shuf"
  LS="BASE,G0,L:trans_fut,L:trans_fut:shuf,AQ:trans_fut:r0.3,AQ:trans_fut:r0.3:shuf"
  bash run_td.sh forward confirm2_fwd $LF 10565-10599; waitfor confirm2_fwd
  bash run_td.sh standard confirm2_std $LS 10565-10599 A,B; waitfor confirm2_std
  bash run_td.sh blocked confirm2_blk $LS 10565-10599; waitfor confirm2_blk
fi

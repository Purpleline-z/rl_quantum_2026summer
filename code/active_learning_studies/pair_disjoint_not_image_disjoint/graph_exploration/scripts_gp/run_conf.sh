#!/bin/bash
# usage: run_conf.sh   (idempotent: cells already written are skipped; restart after a container restart).  Order: confirmation 1 (standard, time-blocked, cold start), confirmation 2 (standard, time-blocked).
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=$PWD:../../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p logs_gp
L=lap3_k10_st4,aw8_all_st2lr0.5,gp_fix3,gp_fix3_noG,gp_fix3_shuf,gp_cvt3
run() { # out seeds extra...
  OUT=$1; SEEDS=$2; shift 2
  for i in 0 1 2 3; do python3 gp_run.py --seeds $SEEDS --out results/gp/$OUT --learners $L --oof --shard $i/4 "$@" >> logs_gp/${OUT}_$i.log 2>&1 & done; wait
}
run conf1 11030-11064 --splits A,B --draws 2
run conf1_blk 11030-11064 --blocked --draws 2
run conf1_cold 11030-11064 --splits A,B --initial random --draws 1
run conf2 11065-11099 --splits A,B --draws 2
run conf2_blk 11065-11099 --blocked --draws 2
echo done > logs_gp/conf.done

#!/bin/bash
# idempotent confirmation chain of the latent-type study (re-run after any container restart).  usage: bash scripts_lt/lt_chain.sh
cd "$(dirname "$0")/../../../new_directions"; export PYTHONPATH=$PWD:../active_learning_program OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p logs_lt
ARMS=lap3_k10_st4,aw8_all_st2lr0.5,lt_aux0.5_em,lt_aux0.1_em,lt_lik_em,lt_pa0.25_em,lt_aux0.5_prior,lt_aux0.1_prior,lt_lik_prior,lt_pa0.25_prior,lt_aux0.5_random,lt_aux0.5_shufprior
BARMS=lap3_k10_st4,aw8_all_st2lr0.5,lt_aux0.5_em,lt_aux0.1_em,lt_lik_em,lt_pa0.25_em,lt_aux0.5_prior
run() { # name seeds splits learners
  for i in 0 1 2 3; do python3 lt_run.py --seeds $2 --splits $3 --learners $4 --out results/lt/$1 --shard $i/4 >> logs_lt/$1_$i.log 2>&1 & done; wait; }
run conf1 10730-10764 A,B $ARMS
python3 rt_a1b.py --seeds 10730-10764 --splits A,B --out results/lt/a1b_conf1 >> logs_lt/a1b_conf1.log 2>&1
run conf1_blocked 10730-10764 blocked $BARMS
run conf2 10765-10799 A,B $ARMS
python3 rt_a1b.py --seeds 10765-10799 --splits A,B --out results/lt/a1b_conf2 >> logs_lt/a1b_conf2.log 2>&1
run conf2_blocked 10765-10799 blocked $BARMS
touch results/lt/CHAIN_DONE

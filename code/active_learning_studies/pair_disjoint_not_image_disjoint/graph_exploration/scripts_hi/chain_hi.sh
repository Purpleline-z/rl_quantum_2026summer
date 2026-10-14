#!/usr/bin/env bash
# Chain: wait for the development run, then confirmation 1, confirmation 2, then the time-blocked runs (development + confirmation).  Idempotent (cells that exist are skipped): safe to restart.
# run from anywhere:  nohup bash scripts_hi/chain_hi.sh > chain_hi.log 2>&1 &
cd "$(dirname "$0")/../../../new_directions"
export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1
LEARN=baseline,aw8,aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4,lap3_k10_st4_shuf,gmc_t_no_rt13
stage() {  # OUT SEEDS DRAWS [extra args]
  OUT=$1; SEEDS=$2; DRAWS=$3; shift 3
  for i in 0 1 2 3; do python3 hi_run.py --seeds "$SEEDS" --out "$OUT" --draws "$DRAWS" --shard $i/4 "$@" >> "$OUT.log.$i" 2>&1 & done
  wait; echo "$(date +%T) done $OUT"
}
while pgrep -f "^python3 hi_run.py --seeds 11300-11319 --out results/hi/dev " >/dev/null; do sleep 20; done
stage results/hi/dev 11300-11319 3            # no-op when finished; completes any missing cell after a restart
stage results/hi/conf1 11330-11364 2 --learners $LEARN
stage results/hi/conf2 11365-11399 2 --learners $LEARN
stage results/hi/dev_blocked 11300-11319 1 --blocked
stage results/hi/conf_blocked 11330-11399 1 --blocked
echo "$(date +%T) chain complete"

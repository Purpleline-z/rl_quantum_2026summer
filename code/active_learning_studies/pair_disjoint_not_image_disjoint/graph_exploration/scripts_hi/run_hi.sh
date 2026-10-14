#!/usr/bin/env bash
# usage: run_hi.sh OUTDIR SEEDS DRAWS [extra hi_run.py args]   (4 shards in the background; idempotent; run from new_directions/)
set -e
cd "$(dirname "$0")/../../../new_directions"
export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1
OUT=$1; SEEDS=$2; DRAWS=$3; shift 3
for i in 0 1 2 3; do nohup python3 hi_run.py --seeds "$SEEDS" --out "$OUT" --draws "$DRAWS" --shard $i/4 "$@" >> "$OUT.log.$i" 2>&1 & done

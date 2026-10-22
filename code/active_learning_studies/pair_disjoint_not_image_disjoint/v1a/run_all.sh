#!/bin/bash
# idempotent launcher: 4 shards over the 35 seeds (finished seeds are skipped); logs in v1a/logs/
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SEEDS=(42 79 123 202 303 $(seq 400 429))
for i in 0 1 2 3; do
  S=""; for j in "${!SEEDS[@]}"; do if [ $((j % 4)) -eq $i ]; then S="$S ${SEEDS[$j]}"; fi; done
  nohup python3 run_v1a.py $S >> logs/run_$i.log 2>&1 &
done

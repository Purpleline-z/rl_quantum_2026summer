#!/bin/bash
# usage: scripts/confirm.sh STRATEGIES   -> runs the 25 CONFIRM seeds (410-429, then 42,79,123,202,303) for both splits, in parallel per split, sequentially over the seed groups.
# Idempotent (finished cells are skipped), safe to relaunch after a container restart.
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
for S in A B; do
  (export JU_SPLIT=$S JU_GRAPH=1 JU_ONLY=$1 JU_MODE=both JU_THREADS=2 PYTHONPATH=../../active_learning_program
   nohup bash -c 'PAIR_STUDY_SEEDS=410-429 python3 judgment_unit_study.py; PAIR_STUDY_SEEDS=42,79,123,202,303 python3 judgment_unit_study.py' > /tmp/ju_confirm_$S.log 2>&1 &)
done

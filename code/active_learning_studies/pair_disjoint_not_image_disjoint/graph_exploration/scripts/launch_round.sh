#!/bin/bash
# usage: scripts/launch_round.sh NAME STRATEGIES SEEDS
#   NAME        label for the log files (/tmp/ju_NAME_{A,B}.log)
#   STRATEGIES  comma list of strategy names (e.g. typed_decisive_coverage_unc,random), see graph_typed.TYPED_NAMES / graph_strategies.GRAPH_NAMES
#   SEEDS       "400-409" (range) or "42,79,123" (list); NOT a mix
# Runs split A and split B in parallel (2 threads each, 4 cores), single-shot + sequential. Finished cells are skipped, so it is safe to relaunch after a container restart.
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
for S in A B; do
  (export JU_SPLIT=$S JU_GRAPH=1 JU_ONLY=$2 JU_MODE=both JU_THREADS=2 PAIR_STUDY_SEEDS=$3 PYTHONPATH=../../active_learning_program
   nohup python3 judgment_unit_study.py > /tmp/ju_$1_$S.log 2>&1 &)
done

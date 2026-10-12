#!/bin/bash
# Round 4 replication (split B, seeds 430-459); idempotent, safe to relaunch after a container restart.
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
for R in 430-444 445-459; do
  (export JU_SPLIT=B JU_GRAPH=1 JU_ONLY=random,typed_decisive_coverage,typed_decisive_coverage_typeonly,typed_decisive_coverage_shuffled,core_set_relation JU_MODE=both JU_THREADS=2 PAIR_STUDY_SEEDS=$R PYTHONPATH=../../active_learning_program
   nohup python3 judgment_unit_study.py > /tmp/ju_rep_$R.log 2>&1 &)
done

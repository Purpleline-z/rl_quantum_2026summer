#!/bin/bash
# usage: phase_run.sh P1|P2|P3  -- relaunch a phase of the overnight run 2 (idempotent: finished cells are skipped); both splits in parallel, 2 threads each.
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
export JU_GRAPH=1 JU_THREADS=2
case "$1" in
  P1) SEEDS=1500-1529; OUT=graph_p1; ONLY=random,typed_decisive_coverage_unc,typed_decisive_coverage,typed_decisive_bald,typed_decisive_coverage_typeonly,typed_decisive_coverage_shuffled,vopt_u,core_set_relation ;;
  P2) SEEDS=600-624; OUT=graph_p2; ONLY=random,vopt_u,gvopt_lap,gvopt_type,gvopt_prop ;;
  P3) SEEDS=1400-1434; OUT=graph_p3; ONLY=${P3_ONLY:?set P3_ONLY to the frozen method list} ;;
  *) echo "unknown phase"; exit 1 ;;
esac
for S in A B; do (nohup ./graph_methods_run.sh $S $SEEDS $ONLY $OUT both > /tmp/ju_${OUT}_$S.log 2>&1 &); done

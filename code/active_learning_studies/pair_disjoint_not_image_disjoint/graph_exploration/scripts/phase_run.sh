#!/bin/bash
# usage: phase_run.sh P1|P2|P3  -- (re)launch a phase of the overnight run 2: 4 processes (2 splits x 2 seed halves), 1 torch thread each; idempotent (finished cells are skipped).
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
export JU_GRAPH=1 JU_THREADS=1
case "$1" in
  P1) LO=1500; HI=1529; OUT=graph_p1; ONLY=random,typed_decisive_coverage_unc,typed_decisive_coverage,typed_decisive_bald,typed_decisive_coverage_typeonly,typed_decisive_coverage_shuffled,vopt_u,core_set_relation ;;
  P2) LO=600; HI=624; OUT=graph_p2; ONLY=random,vopt_u,gvopt_lap,gvopt_type,gvopt_prop,gvopt_sigma,gvopt_lapsigma ;;
  P3) LO=1400; HI=1434; OUT=graph_p3; ONLY=${P3_ONLY:?set P3_ONLY to the frozen method list} ;;
  *) echo "unknown phase"; exit 1 ;;
esac
MID=$(( (LO + HI) / 2 ))
for S in ${SPLITS:-A B}; do
  (nohup ./graph_methods_run.sh $S $LO-$MID $ONLY $OUT both > /tmp/ju_${OUT}_${S}_1.log 2>&1 &)
  (nohup ./graph_methods_run.sh $S $((MID + 1))-$HI $ONLY $OUT both > /tmp/ju_${OUT}_${S}_2.log 2>&1 &)
done

#!/bin/bash
# usage: new_methods_run.sh SPLIT SEEDS ONLY_LIST OUTDIR [MODE]
# Judgment-unit protocol with the re-tuned per-budget head schedule (JUDGMENT_UNIT_RETUNED_RESULTS.md). Cells go to results/new_methods/<OUTDIR>/<SPLIT>/.
# Optional environment: NM_INITIAL_GROUPS=60 (initial set of 60 groups instead of 10); NM_SCHEDULE=old (the single lr 0.01 / 100 steps schedule), NM_INITIAL=random (10 random initial judgments instead of the 10 initial groups).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PAIR_STUDY_SEEDS=$2 JU_INITIAL=${NM_INITIAL:-groups} JU_MODE=${5:-both} JU_THREADS=${JU_THREADS:-2}
export JU_ONLY=$3 JU_RESULTS=new_methods
if [ "${NM_SCHEDULE:-retuned}" = "retuned" ]; then export JU_SCHEDULE=results/pair_endpoint_study/schedule_judgment_unit.json; fi
JU_SPLIT=$1 JU_OUT=$4/$1 python new_methods_study.py

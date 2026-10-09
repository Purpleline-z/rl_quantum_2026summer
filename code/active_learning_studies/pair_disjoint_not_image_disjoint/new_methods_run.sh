#!/bin/bash
# usage: new_methods_run.sh SPLIT SEEDS ONLY_LIST OUTDIR [MODE]
# Judgment-unit protocol with the re-tuned per-budget head schedule (JUDGMENT_UNIT_RETUNED_RESULTS.md). Cells go to results/new_methods/<OUTDIR>/<SPLIT>/.
cd "$(dirname "$0")"
export PAIR_STUDY_SEEDS=$2 JU_INITIAL=groups JU_MODE=${5:-both} JU_THREADS=${JU_THREADS:-2}
export JU_ONLY=$3 JU_RESULTS=new_methods JU_SCHEDULE=results/pair_endpoint_study/schedule_judgment_unit.json
JU_SPLIT=$1 JU_OUT=$4/$1 python new_methods_study.py

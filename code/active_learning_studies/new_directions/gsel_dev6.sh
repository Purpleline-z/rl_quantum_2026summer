#!/bin/bash
# Dev/diagnostic 6 (idempotent): oracle decisive-enrichment upper bound (reads hidden outcomes; diagnostic only), dev seeds, regular + cold. Waits for S2.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_s2r.done ]; do sleep 20; done
SEL=random,tdc,qcov_alloc,ORACLE_tdc,ORACLE_qcov_alloc
for i in 0 1 2 3; do python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 5 --out results/gs_dev6_oracle --shard $i/4 >> logs/gs_dev6_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 5 --out results/gs_dev6_oracle_cold --shard $i/4 >> logs/gs_dev6_cold_$i.log 2>&1 & done; wait
echo done > logs/gs_dev6.done

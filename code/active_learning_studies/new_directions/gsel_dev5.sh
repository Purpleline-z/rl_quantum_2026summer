#!/bin/bash
# Dev screen 5 (idempotent): type-quota coverage (mechanism test of the type allocation), regular + cold. Waits for dev4.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_dev4.done ]; do sleep 20; done
SEL=random,qcov_htr50,qcov_htr50_shuffled,qcov_alloc,qcov_alloc_shuffled
for i in 0 1 2 3; do python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev5 --shard $i/4 >> logs/gs_dev5_$i.log 2>&1 & done; wait
for i in 0 1 2 3; do ND_INITIAL=random python3 gsel_select.py --selectors $SEL --learners aw8_all --seeds 8000-8019 --splits A,B --draws 3 --out results/gs_dev5_cold --shard $i/4 >> logs/gs_dev5_cold_$i.log 2>&1 & done; wait
echo done > logs/gs_dev5.done

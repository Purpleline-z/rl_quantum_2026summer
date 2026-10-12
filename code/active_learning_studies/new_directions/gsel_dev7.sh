#!/bin/bash
# Dev 7 (idempotent): SEQUENTIAL selection (rounds of 10; protocol extension) with the aw8_all learner, dev seeds. Waits for dev6.
cd "$(dirname "$0")"; export PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1; mkdir -p logs
while [ ! -f logs/gs_dev6.done ]; do sleep 20; done
SEL=random,tdc,tdc_typeonly,tdc_shuffled,qcov_alloc,qcov_alloc_typeonly,qcov_alloc_shuffled,vopt_u@aw,htr_vopt_prio@aw,vopt_dec@aw,vopt_dec_typeonly@aw
for i in 0 1 2 3; do python3 gsel_seq.py --selectors $SEL --seeds 8000-8019 --splits A,B --draws 5 --out results/gs_dev7_seq --shard $i/4 >> logs/gs_dev7_$i.log 2>&1 & done; wait
echo done > logs/gs_dev7.done

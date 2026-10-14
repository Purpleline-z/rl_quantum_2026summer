#!/bin/bash
# anchor-far confirmation 1 (seeds 10830-10864), after PREREGISTRATION_AF1.md was committed
cd "$(dirname "$0")"
L=af_base,aw8_all_st2lr0.5,af_pseudoW0.3,af_pseudoW0.3_shuf,af_nbra1,af_nbra1_shuf,af_mixaa0.3,af_mixaa0.3_perm
wait_all() { while pgrep -f "python3 $1 --out" >/dev/null; do sleep 20; done; }
wait_all rt_a1.py   # (a1 launched separately: bash run_rt.sh rt_a1.py af_c1 --seeds 10830-10864 --splits A,B --learners $L --budgets 20,40 --draws 2)
(cd ../../../new_directions; PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1 python3 rt_a1b.py --seeds 10830-10864 --out results/rt/af_c1b)
bash run_rt.sh rt_a3.py af_c1blocked --seeds 10830-10864 --learners $L --budgets 20,40 --draws 2; sleep 30; wait_all rt_a3.py
touch /tmp/af_c1_done

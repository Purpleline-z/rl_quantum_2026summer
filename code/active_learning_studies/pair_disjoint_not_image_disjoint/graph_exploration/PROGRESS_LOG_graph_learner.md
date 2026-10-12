# Progress log — graph-learner branch `claude/graph-learner-20261012` (UTC, 2026-10-12)

- 06:39 start; read COMMON_RULES, wrote seed blocks (7000–7019 dev, 7100–7134, 7200–7234, 7300–7334) into `nd_core.py`; literature notes; `PREREGISTRATION_G0_development_plan.md`.
- 06:48–08:03 dev1: 24 learners (lap λ∈{0.1,0.3,1,3}, rq, SGC, GCN2, cons, shuffled/noprop/noedge controls), 20 seeds × A/B. SGC/GCN encoders and rq alone: no gain; lap: small positive.
- 08:04–08:43 dev2: wider λ grid (to 30), k∈{5,10,20}, `ridge` no-graph control, shuffled controls.
- 08:44–09:31 dev3: noisy-graph dose-response control (`_n1`, `_n3`), lap+rq combinations.
- 09:42–09:51 dev_cold: 10-random-judgment cold start on the dev seeds.
- 09:53 froze 4 candidates + controls + family + success rule (`PREREGISTRATION_G1.md`, `G1_spec.json`), committed (3999406bf) before any confirmation seed was run; launched `scripts_gl/chain_confirm.sh`.
- 11:08 confirmation 1 (7100–7134) finished; success rule applied once by `gl_tables.py confirm`: see `RESULTS_graph_learner.md`.

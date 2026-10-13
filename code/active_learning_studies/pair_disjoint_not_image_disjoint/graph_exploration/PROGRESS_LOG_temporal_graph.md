# Progress log — temporal-graph study (branch claude/temporal-graph-20261013)
- 15:57 start; read rules, SH1, handoffs; parallel branches (graph-mechanism, graph-endpoint) read, red-team RESULTS not present yet (tools rt_*.py only).
- 16:05–16:12 environment: torch from PyPI needs CUDA libs (slow); installed micromamba env `rl` (conda-forge cpu pytorch 2.13, python 3.13) in /home/user/mm; rebuilt SimCLR feature caches (`build_feature_cache.py`, `nd_temporal.py build`). Source with `source /home/user/mm/env.sh`.
- 16:10 `nd_core.py`: seed block TG_* (9200–9219 dev, 9250–9284 confirmation) with assertions; module `nd_temporal_graph.py`, runner `tg_run.py`; baseline equivalence exact (max |d diff| = 0).
- 16:14 development round dev1 launched (12 learners, seeds 9200–9209, A,B, draws 1). Unit tests `test_temporal_graph.py` (10) pass.
- finding while testing: in the red-team blocked split the partner image of a cross-block test group lies outside the held-out blocks; 55–70 % of graph nodes are within 25 frames of some held-out image (the buffer holds only for in-block test images) -> extra `inb_*` stratum.

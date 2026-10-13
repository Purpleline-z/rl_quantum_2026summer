# Progress log — representation study (branch `claude/representation-20261013`, times UTC, 2026-10-13)
- 15:48 start; read COMMON_RULES, RESULTS_selection_htr, PREREGISTRATION_SH1, HANDOFF_selection_htr, RESULTS_graph_learner, nd_graph_learner, nd_core; read what the parallel branches did (not repeated).
- 15:55 network check: huggingface.co, download.pytorch.org, dl.fbaipublicfiles.com answer 403 → no extra pretrained weights; only cached ImageNet-ResNet-18 features exist locally (`image_representation_analysis/.../cache/imagenet_resnet18.npy`, 1278 images).
- 15:59 seed block 8600–8619 / 8650–8684 added to `nd_core.py` (assertions); literature notes; representation learners `nd_representation.py` (representation `s4` reproduces `lap3_k10_st4` exactly, unit test); 7-view SimCLR features (`nd_rep_build.py`).
- 16:00 development plan R0 committed. 16:00–17:10 stage A (12 learners, seeds 8600–8609, splits A and B); ensemble control `rp_ens5` (draw 0).
- 17:10–17:55 stage B1 (graph-free learner on every representation, seeds 8600–8619) and stage B2 (learning-rate x0.5/x2 for baseline and three representation learners; stopped early with 8 complete cells ≈ 4 seeds per split).
- 18:02 PREREGISTRATION_R1 (4 frozen candidates, success rule) committed (c29e696c3) before any confirmation seed; confirmation 1 (8650–8684, splits A,B) launched.

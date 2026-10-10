# Large (GPU) tasks left for the user (written during the autonomous run, 2026-10-09)

The CPU work (graph rules on cached SimCLR features, 512-d, a few hundred images) is done by Claude. The items below need a GPU or are long enough to be worth a Colab session. Order = my guess of value.

1. **Second encoder features for the graph (ImageNet ResNet-18 / DINOv2) for all images.** The graph and type posterior are only as good as the features. Run `build_feature_cache.py`
   adapted to the other encoder for: the 444 images in `simclr_feature_cache.pt` (pool, references, held-out groups) and optionally all 1124 trajectory + 154 ideal images.
   Then rerun `JU_GRAPH=1 python3 judgment_unit_study.py` with the frozen candidate rules (see AUTONOMOUS_RUN_LOG.md for the final names) to see whether the conclusion is encoder-specific.
   (The old representation analysis already has ImageNet ResNet-18 and SimCLR features for 1278 images: `image_representation_analysis/results/representation_exploration/cache/*.npy`; a DINOv2 cache does not exist.)
2. **SimCLR re-pretraining that includes the ~100 new trajectories from Yao** (and all 1124 trajectory frames if not yet used), then rebuild the cache; a better-structured feature space is the most direct way to improve any graph method.
3. **End-to-end fine-tuning with a graph-aware loss** (GCN/GraphSAGE encoder on top of the CNN, per-type Bradley-Terry heads, Laplacian smoothness on unlabeled nodes). This changes the evaluated model (not only the selector), so it is a protocol decision to take with Justin; the frozen-encoder study stays the fixed protocol until then.
4. **Full 33-strategy judgment-unit rerun** (your main rerun, `judgment_unit_study.py` with defaults, both splits) so the graph rules can be compared with all embedding-based rules on the same cells; Random and uncertainty cells for seeds 42-303 and 400-429 already exist in `results/judgment_unit_study/{A,B}_groups`.

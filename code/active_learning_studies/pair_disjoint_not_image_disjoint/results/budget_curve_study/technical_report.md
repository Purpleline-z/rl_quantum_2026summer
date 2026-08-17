# Professor revision technical report

## Protocol

This report is generated from immutable per-job manifests. Results are single-shot acquisitions; performance is reported over budget, while epoch 3 and learning rate 1e-4 are fixed configuration values. Pair partitions are disjoint by unordered pair but may share images.

## Reproducibility and leakage audit

```json
{
  "source_pairwise_csv": "/content/data/original data/Quantum Label Data - Pairwise_Comparisonv1.8.csv",
  "source_pairwise_sha256": "7204dff90c2c93eece551dd81cb8939ff715274063953908104a27ea86f178a5",
  "simclr_checkpoint": "/content/code/classifier 2 context/simclr_resnet18_encoder.pth",
  "simclr_checkpoint_sha256": "076f620d30efc3abb702e16e3590f0776d571ec42ce94309714a123cf80ac25b",
  "valid_pair_rows": 669,
  "unique_unordered_pairs": 179,
  "type_counts": {
    "(\u221a13 x \u221a13)": 157,
    "(1 x 1)": 140,
    "HTR": 137,
    "c(6 x 2)": 120,
    "Twinned(2 x 1)": 115
  },
  "initial_pair_count": 50,
  "candidate_pair_count": 120,
  "selected_pair_count": 75,
  "initial_revealed_row_count": 193,
  "candidate_revealed_row_count": 451,
  "selected_revealed_row_count": 254,
  "exact_pair_overlap_initial_pool": 0,
  "exact_pair_overlap_initial_selected": 0,
  "image_overlap_initial_pool": 28,
  "image_overlap_initial_selected": 19,
  "pairwise_image_overlap_ideal_test": 0,
  "reference_test_overlap": 0,
  "ideal_test_count": 30,
  "ideal_reference_count": 120,
  "pair_disjoint_but_not_necessarily_image_disjoint": true
}
```

## Training

Architecture: SimCLR ResNet-18 encoder with a 512→256→5 Bradley–Terry reward head; all parameters are trainable. Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4, batch size 16, epoch count 3, and the per-job device/seed are stored in `result.json`. Tolerance is not an exposed parameter; trainable layers: full model. Downstream fine-tuning augmentation in `classifier 2 context/train_unified.py` is affine ±5°, translation ±5%, scale 0.95–1.05, and brightness/contrast jitter 0.2. The SimCLR pretraining recipe was not located, so it is not inferred here.

## Results

|   budget | strategy    | symmetry_mode   |   n |   post_test_accuracy_mean |   post_test_accuracy_std |   batch_utility_mean |   batch_utility_std |
|---------:|:------------|:----------------|----:|--------------------------:|-------------------------:|---------------------:|--------------------:|
|       10 | random      | none            |   5 |                  0.413333 |                0.0649786 |          -0.0133333  |            0.101653 |
|       10 | uncertainty | none            |   5 |                  0.386667 |                0.083666  |          -0.04       |            0.17224  |
|       25 | random      | none            |   5 |                  0.433333 |                0.12693   |           0.00666667 |            0.207364 |
|       25 | uncertainty | none            |   5 |                  0.44     |                0.12561   |           0.0133333  |            0.20221  |
|       50 | random      | none            |   5 |                  0.326667 |                0.116428  |          -0.1        |            0.12693  |
|       50 | uncertainty | none            |   5 |                  0.3      |                0.113039  |          -0.126667   |            0.192065 |
|       75 | random      | none            |   5 |                  0.293333 |                0.0434613 |          -0.133333   |            0.141421 |
|       75 | uncertainty | none            |   5 |                  0.38     |                0.140633  |          -0.0466667  |            0.177326 |
|      100 | random      | none            |   5 |                  0.493333 |                0.0760117 |           0.0666667  |            0.176383 |
|      100 | uncertainty | none            |   5 |                  0.646667 |                0.20221   |           0.22       |            0.257768 |

## Historical separation

The `early try 20260728` sequential runs used 70 initial pairs, a 100-pair pool, two-pair acquisition batches, and a 30-image test set. They are exploratory and are not pooled with this controlled single-shot study. The older professor-validation endpoint uses 50 initial pairs, a 120-pair pool, and a 100-pair acquisition.

## Mixture and metadata limits

Feature-space NMF scores are exploratory surrogates: entropy is a multi-component score and residual is an off-basis score, not a measured physical mixture fraction. NMF fits allowed ideal-reference images only and excludes ideal-test images. Metadata fusion activates only when a populated metadata CSV is supplied; the template is `data/metadata/process_metadata_template.csv`.

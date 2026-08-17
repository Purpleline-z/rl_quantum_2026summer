# Process-Metadata Fusion for RHEED Prediction

This folder contains an implementation scaffold, not a completed performance study. It does not launch training or provide Colab/GPU commands.

## Supported session ZIP format

The supplied reference archive contains one session's compact logs:

- `session_metadata.json`: identity and session times;
- `sensor_log.csv`: timestamped monitor values;
- `heartbeat_log.csv`: timestamped RHEED frame references;
- optional event, commit, and growth-note logs, which are deliberately excluded from model inputs.

It contains **no image files**. The `frame_path` values are laboratory Windows paths, not a trustworthy mapping to repository images. Before any experiment, create an explicit CSV with exactly these columns:

```text
source_frame_path,resolved_image_path
```

The source path is matched after slash/case normalization. Every resolved image must exist. The implementation never guesses a filename correspondence.

## Registered features and architecture

The feature vector consists of elapsed time, pyrometer temperature, actual Mistral voltage/current, chamber pressure, and causal temperature/voltage/current rates. Each of the eight values receives a missingness indicator. Image-plus-metadata fusion is:

```text
SimCLR image embedding (512) + process MLP embedding (64)
→ concat (576) → Linear(576, 512) + ReLU → existing five-output reward head
```

The study also defines image-only and metadata-only ablations. It does not implement DAFT, FiLM, cross-attention, text modeling, or a new acquisition strategy.

## Safety checks

The validators stream each ZIP directly, align each heartbeat frame to the latest sensor row at or before its timestamp, record sensor lag and unresolved image paths, and reject label-like columns. Training is intentionally blocked unless an explicit mapping resolves every frame and at least three sessions support mutually run-disjoint and image-disjoint train/validation/outer-test partitions.

The reference ZIP has one session, so it should pass archive/alignment validation and fail the capacity check. That is the expected result.

## Literature

- Du et al. describe tabular-image learning with explicit treatment of incomplete tabular data: [TIP (2024)](https://arxiv.org/abs/2407.07582).
- Wolf, Pölsterl, and Wachinger compare concatenation baselines with conditional image-tabular fusion: [DAFT (2022)](https://doi.org/10.1016/j.neuroimage.2022.119505).
- Salvi et al. review multimodal data integration, preprocessing, and validation concerns: [Information Fusion (2024)](https://doi.org/10.1016/j.inffus.2023.102134).


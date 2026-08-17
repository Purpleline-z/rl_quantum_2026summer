# Process-Metadata Fusion for RHEED Prediction

## Question

Can causally available, numerical growth-monitor metadata add predictive value to a SimCLR-based RHEED image model without leaking labels, run identity, future measurements, or test information?

## Observed session-bundle interface

The reference bundle `AI_AJ002_STO_20260805_165051.zip` is a small, metadata-only record of one growth session. `sensor_log.csv` records a timestamp, elapsed time, pyrometer temperature, actual Mistral voltage/current, and chamber pressure. `heartbeat_log.csv` records timestamped laboratory `frame_path` references. It contains no RHEED image bytes. The path references must be explicitly mapped to actual image files before an experiment is possible.

The registered feature order is elapsed time; pyrometer temperature; actual voltage; actual current; chamber pressure; and backward-looking temperature, voltage, and current rates. Missingness is explicit. At each image time the implementation uses only the latest sensor row at or before that timestamp; it does not interpolate from future measurements.

## Why this architecture

The primary model has an image branch (shipped SimCLR ResNet-18, 512 dimensions) and a process branch (16 values after missingness indicators, encoded as `16 -> 64 -> 64`). The embeddings are concatenated and projected from 576 to 512 dimensions before the existing five-output Bradley--Terry reward head. This late-fusion design is deliberately the first architecture because it is the requested separate-branch/concatenation method, adds limited capacity, and makes image-only, metadata-only, and fused-model comparisons interpretable.

Recent tabular-image work identifies heterogeneous and missing tabular values as a central practical problem; it motivates explicit missingness representation and coverage audits rather than false physical zeros ([Du et al., 2024, TIP](https://arxiv.org/abs/2407.07582)). Concatenating a high-level image descriptor and a tabular representation is an established baseline for image-plus-tabular prediction. DAFT instead adds conditional affine modulation inside a CNN; it is a future sensitivity direction, not part of this small-data first study ([Wolf, Pölsterl, and Wachinger, 2022](https://doi.org/10.1016/j.neuroimage.2022.119505)). A 2024 review likewise emphasizes heterogeneous sources, preprocessing, and fusion validation in multimodal systems ([Salvi et al., 2024](https://doi.org/10.1016/j.inffus.2023.102134)).

## Data and leakage policy

Each session ZIP is one growth run. The primary eventual split is mutually run-disjoint and image-disjoint train, validation, and outer-test partitions. The code refuses to relax either rule. One session cannot form these partitions, so the supplied ZIP is valid only for interface/audit testing. A later experimental dataset needs at least three image-resolved session bundles and sufficient class/pair capacity in each partition.

The first study excludes reconstruction columns, classifier outputs/status, corrected labels, live labels, notes in `growth_log.xlsx`, manual-event content, auto-capture decisions, and set-change outcomes. Those sources are not neutral process measurements and would compromise a clean measurement of metadata fusion.

## Use in the technical report

This study currently contributes only a registered data interface, architecture, and safety contract. It is not used in any accuracy, active-learning, or physics claim in the technical report. It becomes performance evidence only after multiple session bundles, verified image mappings, leakage audits, validation-only model selection, and untouched run-disjoint/image-disjoint outer-test evaluation.


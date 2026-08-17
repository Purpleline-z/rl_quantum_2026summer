# Process-Metadata Fusion: Data Interface and Architecture Guide

## Research purpose

Growers use elapsed time, temperature, voltage, current, and chamber pressure when interpreting a growth trajectory. This study tests whether those same measurements provide information that complements a RHEED image model. The first endpoint is classifier/reward-model performance; active-selection effects are a later, separate test.

The implementation is intentionally ready before the full data are available. It can validate session-log bundles, align monitor values causally to frame timestamps, verify image mappings, and reject unsafe data. It does not start a training experiment with the single supplied bundle.

## Observed bundle format

The reference ZIP contains one growth session, 198 `sensor_log.csv` records, and 132 `heartbeat_log.csv` frame references. It contains no images. A usable multi-session dataset must provide one or more bundles with the same required members plus an explicit mapping table:

```text
source_frame_path,resolved_image_path
```

The mapping is a scientific provenance record, not a filename convenience. It establishes which experimental image received which monitor values. The validator normalizes path spelling, confirms the resolved file exists, and records unresolved paths instead of guessing.

## Features and timing

The process vector contains elapsed time, pyrometer temperature, actual Mistral voltage/current, chamber pressure, and backward-looking temperature/voltage/current rates. Each numerical feature has a missingness flag, giving 16 model inputs. A frame at time `t` receives only the latest sensor value recorded at or before `t`; no future interpolation is permitted.

This timing rule makes the model suitable for a future next-action setting. It also prevents a deceptively high retrospective score from using a temperature or pressure change that occurred after the RHEED frame.

## Models compared

| Arm | Purpose |
|---|---|
| Image-only | Current shipped SimCLR image model baseline. |
| Metadata-only | Detects whether monitor traces alone carry useful state information or reveal a split shortcut. |
| Late fusion | Encodes image and metadata separately, concatenates their representations, and uses the existing five-output reward head. |

Late fusion uses a 512-dimensional SimCLR image embedding, a `16 -> 64 -> 64` metadata MLP, a `576 -> 512` fusion layer, and the current Bradley--Terry reward head. It is the first architecture because it is transparent and directly comparable with both single-modality controls. DAFT/FiLM, attention fusion, text notes, and event features are reserved for later registered comparisons.

## What is excluded and why

Reconstruction labels, classifier outputs, grower corrections, live labels, notes, post-hoc commits, auto-capture states, and set-change outcomes are excluded. They may be useful for future decision-support studies, but they would not answer whether routine monitor values improve image interpretation. [study_description.md](study_description.md) gives the complete rationale and literature context.

## Readiness and next decision

The present one-session bundle passes parsing and causal alignment, but every frame is unresolved and one run cannot create independent train, validation, and outer-test partitions. The implementation therefore blocks training. The next required input is at least three image-resolved sessions for a technical split, with a target of at least twelve independent sessions for a preliminary comparison. Once supplied, the first result will be fusion versus image-only on a run-disjoint and image-disjoint real outer test.

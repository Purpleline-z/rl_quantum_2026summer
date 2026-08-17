# Process-Metadata Fusion for RHEED Prediction

## Purpose and scope

This study asks a narrow question: **do routine process-monitor measurements, available no later than a RHEED frame, improve a RHEED prediction model beyond the image alone?** The first endpoint is the downstream reconstruction classifier/reward model, not active selection. A second, separate phase will test whether a verified classifier benefit improves active-learning acquisition. A better classifier is not automatically a better selector.

The image branch is the shipped SimCLR ResNet-18 and the existing five-output Bradley--Terry reward head for `(1 x 1)`, `Twinned(2 x 1)`, `c(6 x 2)`, `(sqrt(13) x sqrt(13))`, and `HTR`. The process branch supplies context that growers already use in practice: elapsed time, temperature, voltage, current, chamber pressure, and selected rates. The principal risk is shortcut learning: the model could learn an experiment/run identity, a future intervention, or a label embedded in a log instead of learning an image-plus-physics relationship. The implementation therefore prioritizes causal alignment, strict session splits, and explicit negative controls.

This folder is registered infrastructure and data-interface documentation. It is **not** a completed performance study and contributes no metadata performance claim to the technical report.

## What the supplied ZIP establishes

`AI_AJ002_STO_20260805_165051.zip` is a 179 KB metadata bundle for one growth session. It contains no BMP, PNG, or JPEG images.

| Member | Observed content | Permitted role |
|---|---|---|
| `session_metadata.json` | sample ID, grower, chamber, session start/end, row counts | Defines session/run provenance. |
| `sensor_log.csv` | 198 timestamped monitor rows at roughly 10-second intervals | Primary numerical source. |
| `heartbeat_log.csv` | 132 timestamped `frame_path` references | Associates a frame time with prior sensor data. |
| `set_change_events.csv` | voltage/current changes | Audit-only in this study. |
| `auto_capture_events.csv` | capture workflow and change events | Audit-only in this study. |
| `commit_log.csv` | reconstruction values, classifier outputs/status, notes | Explicitly prohibited as input. |
| `live_labels.csv` | live reconstruction labels | Explicitly prohibited as input. |
| `growth_log.xlsx` | human-entered summaries and notes | Excluded from the first numerical study. |

The observed primary monitor columns are `elapsed_s`, `pyrometer_temp_C`, `mistral_v_actual_V`, `mistral_i_actual_A`, and `chamber_pressure_mbar`. Frame references use laboratory Windows paths such as `E:\\ChMBE\\2026\\AI_AJ002_STO_20260805_165051\\frames\\heartbeat_001_165106.bmp`.

A path reference is not proof that an image exists in this repository. Before an experiment, the laboratory/project must supply an explicit mapping CSV with `source_frame_path,resolved_image_path`. The adapter normalizes slash and case spelling, requires each resolved file to exist, and never guesses a match from a basename or timestamp.

The supplied ZIP passed the schema and causal-alignment check: 198 sensor rows, 132 heartbeat references, zero future sensor matches, and 132 unresolved image references. The unresolved count is expected because neither image bytes nor a mapping are present.

## Registered representation and causal alignment

Each heartbeat frame at time `t` receives the most recent sensor row with timestamp `<= t`, using a backward `merge_asof` join. The derived manifest records frame timestamp, sensor timestamp, lag, session/run ID, archive SHA-256, normalized source path, resolved-image status, the feature values, and missingness. Future readings are never interpolated or backfilled.

The fixed first-study feature order is:

1. frame `elapsed_s`;
2. `pyrometer_temp_C`;
3. `mistral_v_actual_V`;
4. `mistral_i_actual_A`;
5. `chamber_pressure_mbar`;
6. backward-looking `d(pyrometer_temp_C)/dt`;
7. backward-looking `d(mistral_v_actual_V)/dt`;
8. backward-looking `d(mistral_i_actual_A)/dt`.

Rates use a sensor row and its immediately preceding sensor row; the first rate is missing. Every value has a missingness indicator, producing a 16-dimensional model input. Median imputation and standardization are fitted on training sessions only; validation and outer-test sessions cannot affect centers, scales, or imputation values.

The study does not silently add every available sensor channel. The registered variables are the five quantities identified by the laboratory as most routinely used by growers, plus the three requested rates. A small predeclared set is more interpretable and less likely to become a disguised session identifier. Other temperatures, powers, setpoints, and event variables may be evaluated only in a later explicitly registered ablation.

## Model architecture and rationale

The primary late-fusion model is:

```text
RHEED image -> shipped SimCLR ResNet-18 -> 512-dimensional image embedding
process metadata -> MLP 16 -> 64 -> 64 with ReLU
concatenate 512 + 64 -> Linear(576, 512) + ReLU
unchanged five-output Bradley--Terry reward head
```

It is compared to two essential controls:

- **image-only:** the existing shipped SimCLR image model and five-output head;
- **metadata-only:** the 16-value process vector passed through its own MLP and five-output head.

The metadata-only arm is a shortcut diagnostic. Strong metadata-only performance may indicate a real process-state signal, but it may also indicate inadequate session separation. It is not acceptable to report fusion without these two controls.

Late fusion is the initial choice because it is the requested separate-branch/concatenation design, adds limited capacity to a small labeled dataset, and makes modality contributions auditable. Recent tabular-image work identifies heterogeneous and missing tabular inputs as a central practical problem, supporting missingness indicators and data-quality audits ([Du et al., 2024, TIP](https://arxiv.org/abs/2407.07582)). Concatenated representations are an established baseline. DAFT is a more complex alternative that conditions late CNN features on tabular information; it is deferred until the simple architecture has been tested with adequate data ([Wolf, Pölsterl, and Wachinger, 2022](https://doi.org/10.1016/j.neuroimage.2022.119505)). A recent multimodal review likewise emphasizes preprocessing, heterogeneous data sources, and careful validation ([Salvi et al., 2024](https://doi.org/10.1016/j.inffus.2023.102134)).

## Leakage policy and excluded information

The initial scientific question is whether routine monitor measurements complement the image. The following sources are deliberately excluded because they are labels, model outputs, annotations, post-hoc actions, or free text:

- `recon_*`, `classifier_recon_*`, `classifier_status`, and `grower_corrected` in `commit_log.csv`;
- reconstruction content in `live_labels.csv`;
- notes, operations, and free text in `growth_log.xlsx`, `commit_log.csv`, and `manual_events.csv`;
- auto-capture decisions, `event_state`, `state_changed_at`, and image change scores;
- set-change events and their outcomes.

This does not mean those records are scientifically unhelpful. They may support a later decision-support, text, or intervention-modeling study. They are not admissible for the present question because they would make a metadata improvement impossible to interpret. The implementation rejects unregistered or label-like feature names and retains archive/join provenance for audit.

## Split policy, data requirements, and later evaluation

Each session ZIP is one `run_id`. The intended future split is complete-run-disjoint and image-disjoint train/validation/outer-test partitions. Pairwise training groups and active-learning candidates may only come from training runs. Validation selects hyperparameters. Outer test remains unseen until final evaluation.

Run-disjointness is essential: frames within one growth share a correlated temperature, voltage, current, and pressure trajectory. Holding out only individual images could allow the metadata branch to identify the same experiment in train and test. The relevant question is generalization to a new growth session.

One session is enough to validate the interface but cannot produce train, validation, and outer-test partitions. The code refuses to run without at least three image-resolved sessions. Three is a technical minimum, not a persuasive sample. The requested preliminary target is **at least 12 independent image-resolved growth sessions**, ideally with usable frames across relevant reconstruction classes in every split. More sessions are needed if one session dominates a class or multiple sessions follow nearly identical recipes.

Once adequate data exist, the classifier phase will use paired seeds, validation-only protocol selection, raw per-seed results, per-class metrics, missingness and lag summaries, and paired confidence intervals. A metadata claim requires a stable fusion advantage on the untouched run-disjoint and image-disjoint outer test. Only after that will a separately reported active-learning phase compare image-only and fusion-enabled budget curves under approved selectors. Candidate preference labels remain hidden from the selector; visible metadata must be causally available at frame time.

## Current status and comparability limits

The current files implement archive validation, causal alignment, explicit image mapping, run/image-disjoint capacity checks, the three model arms, and behavior tests. They do not start a CPU or GPU experiment. No raw ZIP, laboratory image, mapping, checkpoint, or performance result is committed.

This study must not be pooled with historical pair-disjoint active-learning curves, the Classifier2 report, simulator augmentation, or trajectory-ordering analyses. It has a different modality, requires stricter session-level separation, and has not yet received an image/session mapping. Any future comparison must state exactly which sessions, images, labels, initialization, training protocol, and endpoint are shared.

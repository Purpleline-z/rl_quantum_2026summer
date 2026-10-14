# FT0 — development plan: CPU partial fine-tuning of the SimCLR encoder (branch `claude/finetune-cpu-20261014`, dev seeds 10300–10319)

Written before the first development run. Everything tried is logged in `FT_DEV_LOG.md` (append-only).

## Question
Earlier work swapped frozen representations (null) and tuned the head (plateau). Does *adapting* the encoder (layer 4, on cached layer-3 maps) change held-out pair-group AUC/accuracy, and in particular the anchor-far stratum (> 0.03 cosine distance from every typed ideal image), where nothing has helped so far?

## Compute facts (measured, 1 thread, this container: 4 cores, AMX/bf16)
layer 4 forward+backward ≈ 20 ms/image fp32, 7.3 ms bf16 (forward only 5.7 / 1.6 ms); layer-3 cache 1278 × 256 × 14 × 14 fp32 = 257 MB (not committed; `python3 nd_finetune.py build`, 20 s). Full-recipe fine-tuning per fit (all 700+ images per step) is infeasible (≈ 5 s/step × hundreds of steps × 840 fits); hence mini-batch designs below.

## Two families (both identity-safe)
* **P — per-fit refinement** (`ftp_*`, `nd_finetune.py`): stage 1 = incumbent head on frozen features (= `lap3_k10_st4`, LP-FT analogue); stage 2 = `steps` mini-batch steps of head + layer 4 on pair loss + anchor loss (aw 8, all ideal images, 12 per class per step) + Laplacian (λ 3, 24 edges per step) [+ consistency] with L2-SP, feature-map augmentation (flip, ±1 shift, intensity jitter), WiSE interpolation option. Matched control `enc_lr=0`: the identical stage 2 with the encoder frozen. Graph: frozen (cached features), current (rebuilt from the adapting embedding every 20 steps) or shuffled (control). Budgets 20 and 40 only (≈ 70 s per fit); budgets 10/60 are not run (stated in the results).
* **S — shared label-free encoder** (`fts_*`, `ft_shared.py`): layer 4 adapted ONCE on images that occur in no labelled pair group (and no content twin of one) plus the typed ideal images; no pair labels; terms: anchors (aw 8), Laplacian on the kNN graph of the adaptation images, augmentation consistency, InfoNCE between two augmented views, L2-SP. The incumbent recipe then runs on the adapted features (graph from adapted or original features). All budgets 10/20/40/60, 3 draws are affordable (≈ 6 s per fit). Time-blocked split: a shared encoder has seen non-group frames inside the 25-frame buffer of the held-out blocks, so the blocked split is evaluated only for family P (per-seed contexts exclude the buffer zone) and, for S, with a per-seed encoder if time allows.

## Development screens (≤ 10 variants logged; every variant run is listed in `FT_DEV_LOG.md`)
Round 0: does stage 2 itself disturb the head (frozen control vs incumbent)? Round 1 (P): enc lr, L2-SP, last block only, augmentation on/off, head-lr multiplier, steps, current vs frozen graph, anchor-only vs pair+anchor, consistency, WiSE. Round 2 (S): anchors + graph (A), +consistency, +InfoNCE, shuffled-graph control, lr / steps.
Endpoints in development: paired gain over `lap3_k10_st4` (AUC, accuracy, anchor-far AUC/accuracy, log-loss), and over the matched frozen control; development p-values are not used for any claim.

## Freeze (at ≈ 4.5 h)
≤ 4 candidates with their controls, the test family and the success rule will be written to `PREREGISTRATION_FT1.md` / `FT1_spec.json` and committed BEFORE any confirmation seed (10330–10364, 10365–10399) is run.

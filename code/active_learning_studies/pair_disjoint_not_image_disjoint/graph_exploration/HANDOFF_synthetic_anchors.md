# Handoff — synthetic-anchor study (branch `claude/synthetic-anchors-20261014`, cut from `claude/integration-20261013`)

## State
Finished: reading, simulator + calibration, 5 development rounds (seeds 11100–11119, ≈ 45 learners) + zero-pair-label diagnostics, frozen `PREREGISTRATION_SA1.md` (commit 62648d287, before any confirmation cell), confirmation 1 (11130–11164) and confirmation 2 (11165–11199) incl. the red-team blocked split. Results: `RESULTS_synthetic_anchors.md` (generated), `SUMMARY_ZH_synthetic_anchors.md`, `GPU_TASKS_synthetic_anchors.md`, `notes_synthetic_anchors_literature.md`, `SA1_dev_tables.md`, `PROGRESS_LOG_synthetic_anchors.md`.

**Result in one line:** simulator images as typed anchors do not help (confirmed twice; a moment-matched Gaussian control is equal); the anchors' image style does matter: the gate between the baseline head and a head trained with direct-beam-masked, rescaled ideal images improves held-out AUC (+0.003…+0.005, all four tests Holm-significant over the two confirmations) and log-loss (+0.008) but its accuracy gain does not clear Holm, so no candidate meets the pre-registered rule.

## Seeds (do not reuse)
Block 11100–11299: development 11100–11119, confirmation 1 11130–11164, confirmation 2 11165–11199 (all used); 11120–11129 and 11200–11299 are unused (this study's own follow-ups only). `nd_core.py` has the `SA_*` lines with assertions.

## Code map (`new_directions/`)
- `rheed_sim.py`: kinematic renderer (`generate`, `render`, `rods`, `TUNE` = calibrated multipliers, `TUNE0` uncalibrated, `TUNE_T` per-type); `syn_calibrate.py` (global calibration against unlabelled frames + masked ideal images), `syn_calibrate_types.py` (per type, over-fits: see results), `syn_bank.py` (render + encode a bank, `build_augreal`, `align_bank` mean / coral / typemean, `build_tmgauss`), `syn_gap.py` (gap report), `syn_ideal_fix.py` (fix16 / mask / mask_keep16 variants of the 154 ideal-image features), `syn_graph_stats.py`.
- `nd_synthetic.py`: `fit_syn` (reproduces `lap3_k10_st4` exactly without additions; options `ideal`, `ideal_graph`, `shuf_graph`, `drop16`, `extra`, `syn` incl. `perm` / `replace` / `nodes`; fits are cached per (spec name, seed, labelled set) inside a context), two-head learners (`register_two`, modes `mean` and `gate` with the anchor-proximity gate `anchor_distance_fn`), registry `LEARNERS` / `SPEC_OF`; `syn_specs.py` (all development and frozen learners), `syn_run.py` (runner incl. blocked split), `syn_anchor_only.py` (zero-pair-label diagnostic).
- `graph_exploration/`: `sa_tables.py` (dev / confirm with Holm, verdicts, far family, controls on far; copy of `tg_tables.py`), `sa_report.py` + `results_narrative_synthetic_anchors.md` -> `RESULTS_synthetic_anchors.md` (edit the narrative, never the generated file), `scripts_sa/{run_sa.sh,chain_confirm.sh}`, `SA1_spec.json`.
- Tests: `code_behavior_tests/test_synthetic_anchors.py` (12): seed block, detector geometry, lattice areas, Laue circle through the specular spot, determinism, 16-bit ideal images (18), aligned banks, exact baseline equivalence, weighted anchor loss, gate uses image features only, calibration excludes every labelled pair-group image.
- Results: `new_directions/results/syn/{dev1..dev5,anchor_only*,confirm1,confirm1_blocked,confirm2,confirm2_blocked,confirm12 (pooled copy),gap_*.md,bank_*.pt (force-added, small),ideal_fixed.pt}`.

## Environment pitfalls
- No torch in the image; use the micromamba env (`/home/user/mm/env.sh`, conda-forge CPU torch; see `HANDOFF_temporal_graph.md`). The container restarted twice during this study (background jobs die, files persist): every runner is resumable; re-launch the same command.
- `pkill -f` / `pgrep -f` with a pattern that occurs in your own command line kills your shell; use `ps -eo pid,args | grep "[p]attern"` then kill by pid.
- Do not start two runners on the same output directory (read-modify-write race on the cell JSONs).
- Banks / ideal-feature files are produced with torch 2.13 CPU; the committed `.pt` files make the learners reproducible without re-rendering.

## Findings worth keeping
1. 18 of 154 ideal images (10 HTR, 8 RT13) are 16-bit PNGs that the loader clips (≈ 88 % saturated pixels): tell the lab; `Image.open(p).convert("L")` is the culprit (`syn_ideal_fix.load_fixed` shows the 255/65535 rescale).
2. (1x1)/c(6x2) ideal images are uncropped (direct beam) and sit 0.3–0.7 from every trajectory frame; masking the rows above the shadow edge moves them to 0.17–0.33.
3. Zero-pair-label diagnostic: real anchors alone give held-out AUC 0.92–0.94; synthetic anchors alone 0.58–0.60 (permuted control 0.41–0.45); type-mean-aligned synthetic 0.90–0.91, a moment-matched Gaussian the same.
4. Gate / ensemble: see RESULTS §6.

## What I would do next
1. Re-run the gap table and the zero-label diagnostic with the lab's existing 2250-image simulator archive (not available here) — the only way to test a *better* simulator.
2. Decide the intended ideal-image preprocessing with the lab (GPU_TASKS item 1) and re-run the confirmed ladder with it as the new baseline (the gate's gain is the gain of doing this at the head level only where it pays).
3. A pre-registered 35-seed test of the gate's AUC/log-loss as the primary endpoint (accuracy secondary) on fresh seeds 11200+.

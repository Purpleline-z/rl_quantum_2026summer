*Point estimates only (intervals, seeds and evidence in the full table below / `claims_table.md`). Splits A / B.*

| ID | Claim | Status | ΔAUC A / B | Δacc A / B | Anchor-far ΔAUC A / B |
|---|---|---|---|---|---|
| L1 | aw8 (anchor weight 0.25 -> 8) beats the default head | CONFIRMED + replicated | +0.012 / +0.019 | +0.019 / +0.023 | +0.050 / +0.015 |
| L2 | aw8_all (all ideal images as anchors) beats the default head | CONFIRMED | +0.020 / +0.023 | +0.026 / +0.030 | +0.044 / +0.035 |
| L2b | aw8_all vs aw8 | DESCRIPTIVE | +0.007 / +0.009 | +0.009 / +0.009 | -0.006 / +0.020 |
| L3 | Laplacian kNN-graph regulariser (lap3_k10) beats aw8_all | CONFIRMED + replicated | +0.015 / +0.014 | +0.014 / +0.013 | +0.016 / +0.016 |
| L4 | 4x training steps (lap3_k10_st4) beats lap3_k10 | CONFIRMED + replicated | +0.010 / +0.007 | +0.013 / +0.014 | -0.006 / +0.005 |
| L5 | BEST: lap3_k10_st4 beats the tuned graph-free control | CONFIRMED + replicated | +0.021 / +0.016 | +0.023 / +0.020 | +0.016 / +0.021 |
| L6 | whole recipe (lap3_k10_st4) vs aw8_all | DESCRIPTIVE | +0.027 / +0.022 | +0.031 / +0.022 | +0.009 / +0.020 |
| L7 | graph-free schedule tuning alone vs aw8_all | DESCRIPTIVE | +0.006 / +0.006 | +0.008 / +0.003 | -0.007 / -0.001 |
| L8 | real graph beats shuffled-graph control (a low bar) | PRE-REG. control | +0.085 / +0.089 | +0.120 / +0.115 | not evaluated |
| L9 | RT13-off variant beats lap3_k10 (not tested vs st4) | CONFIRMED + replicated | +0.010 / +0.010 | +0.007 / +0.006 | not evaluated |
| L10 | uniform lambda 10 beats lap3_k10 | NOT CONFIRMED | +0.004 / +0.002 | +0.004 / +0.003 | not evaluated |
| L11 | Laplacian + type-posterior term beats aw8_all | CONFIRMED (partly / not robustly replicated) | +0.012 / +0.007 | +0.021 / +0.013 | not evaluated |
| L12 | GP learner beats the anchor-weighted MLP head | NOT CONFIRMED | +0.013 / +0.015 | -0.005 / -0.004 | not evaluated |
| L13 | aw8_all improves HTR vs default head | NOT CONFIRMED | HTR acc: +0.015 / +0.032 | HTR ll: +0.004 / +0.066 | not evaluated |
| D1 | long schedule does not help graph-free aw8_all | DEVELOPMENT only | +0.001 / -0.005 | +0.003 / -0.000 | not evaluated |
| D2 | graph-free consistency regulariser comes close (cons - st4 shown) | DEVELOPMENT only | -0.005 / -0.007 | -0.021 / -0.018 | not evaluated |
| G1 | gain tracks graph quality (noise dose-response; rho) | CONFIRMED + replicated | AUC rho +0.76 / rho +0.67 | acc rho +0.64 / rho +0.65 | not evaluated |
| G2 | 10% random edge rewiring destroys the gain; local rewiring harmless | DEVELOPMENT only | -0.039 / -0.038 | -0.063 / -0.055 | not evaluated |
| G3 | shuffled graph is far worse than no graph (lambda 10) | PRE-REG. control | -0.128 / -0.126 | -0.129 / -0.125 | not evaluated |
| G4 | more unlabelled frames: gain saturates at ~10% of frames | PRE-REG. rule met, practical claim not supported | +0.013 / +0.012 | +0.014 / +0.009 | not evaluated |
| G5 | harsh splits (held-out run / temperature; fixed test sets): gain kept | CONFIRMED + replicated | +0.020 / +0.018 | +0.015 / +0.025 | not evaluated |
| G6 | paper's type-accuracy endpoint: no graph advantage | PRE-REG. rule: mixed (see full table) | n/a (accuracy endpoint) | GE1 +0.007 (Holm 1.00); GE2 +0.026 (Holm 0.004) | not evaluated |
| S1 | vopt_u beats random with the default head | CONFIRMED (partly / not robustly replicated) | +0.010 / +0.013 | +0.016 / +0.022 | not evaluated |
| S2 | vopt_u beats random with the aw8 head | NOT CONFIRMED | +0.002 / +0.005 | +0.000 / +0.002 | not evaluated |
| S3 | 4 selection rules beat random on the lap3_k10 learner | NOT CONFIRMED | +0.001 / -0.001 | +0.003 / -0.002 | not evaluated |
| S4 | HTR-directed selection (vopt_htr4) | PRE-REG. rule: mixed (see full table) | HTR acc: +0.005 / +0.006 | HTR ll: +0.026 / +0.019 | not evaluated |
| S5 | HTR: st4 vs tuned graph-free control | PRE-REG. rule: mixed (see full table) | HTR acc: +0.014 / +0.014 | HTR ll: +0.037 / +0.022 | not evaluated |
| S6 | HTR: st4 vs lap3_k10 | PRE-REG. rule: mixed (see full table) | HTR acc: +0.017 / +0.019 | HTR ll: +0.124 / +0.120 | not evaluated |
| R1 | gain concentrated in anchor-near judgments | AUDIT (red-team dev) | +0.019 / +0.017 | near<=0.001: +0.035 / +0.028 | +0.016 / +0.021 |
| R2 | aw8 gain is not an anchor-copy artefact | AUDIT (red-team dev) | +0.013 / +0.022 | near<=0.001: +0.004 / +0.028 | +0.044 / +0.035 |
| R3 | time-blocked split keeps the gain | AUDIT (red-team dev) | +0.026 | +0.012 | not evaluated (blocked split has no anchor strata) |

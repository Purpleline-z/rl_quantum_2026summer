# Progress log — transductive / forward-in-time study (branch claude/transductive-20261014, seeds dev 10500–10519, confirm1 10530–10564, confirm2 10565–10599)

## Stage 0 (setup)
- Fetched `claude/integration-20261013`, cut `claude/transductive-20261014`. Container had no torch: micromamba env `rl` (conda-forge CPU torch) in `/home/user/mm` (`env.sh`, not in git); caches rebuilt (`build_feature_cache.py` 18 s, `nd_temporal.py build` 16 s; 1278 images).
- One lap3_k10_st4 fit takes ≈ 3 s on one core (the temporal-graph handoff says ≈ 10 s: faster machine); confirmations are cheap.
- `nd_core.py`: TD seed block added with assertions. `td_core.py` (node-set modes, forward contexts, `fit_td`, score propagation), `td_learners.py` (named learners), `td_run.py` (runner). `fit_td` reproduces `lap3_k10_st4` exactly (max |Δd| = 0, tested in the smoke run).
- Data facts: 168 groups, 284 images, all images are trajectory frames of three runs (RR220204A / RR220206A / RR220411A, folder dates 2022-02-04, -02-06, 2022-04-11), so every group can be assigned a time (group time = time of its later image, calendar rank of run first); 104 of 168 are cross-run pairs (calendar order puts a group at its later run), 29/35/104 groups have group time in the three runs.
## Protocol design decisions (see PREREGISTRATION_TD0_development_plan.md)
- Forward cuts: n_past ∈ {12, 24, 40, 54, 70, 90} calendar-ordered groups (two per run; ±2 jitter by seed), FA = next 40 groups, FB = following 34 groups (the test sizes of Splits A and B); training groups sharing an image with a test group are dropped (identity rule), ≈ 8–78 groups kept, 25–237 judgments.

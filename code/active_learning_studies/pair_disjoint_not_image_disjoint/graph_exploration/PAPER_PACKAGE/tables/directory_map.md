| dir | seeds | n_seeds | n_learners | description |
|---|---|---|---|---|
| confirm1 | 3000-3034 | 35 | 3 | ND1 confirmation: default head / aw8 / aw8_all x random / vopt_u (PREREGISTRATION_1) |
| confirm3_cold | 4000-4034 | 35 | 2 | ND3 cold-start confirmation (10 random initial judgments) |
| confirm4_htr | 5000-5034 | 35 | 3 | ND4 HTR confirmation (baseline / aw8 / aw8_all) |
| confirm5_gp | 6000-6034 | 35 | 3 | ND5 confirmation of the GP preference learner |
| ge/conf1_class | 8500-8534 | 35 | 8 | GE classification endpoint (28 outer-test ideal images) |
| ge/conf1_dose | 8500-8534 | 35 | 10 | GE dose-response of unlabelled frames |
| ge/conf1_harsh | 8500-8534 | 35 | 4 | GE harsh splits (held-out run / temperature tercile; contains .d.npz per-judgment scores) |
| ge/conf2_class | 8550-8584 | 35 | 8 | GE classification endpoint (28 outer-test ideal images) |
| ge/conf2_dose | 8550-8584 | 35 | 10 | GE dose-response of unlabelled frames |
| ge/conf2_harsh | 8550-8584 | 35 | 4 | GE harsh splits (held-out run / temperature tercile; contains .d.npz per-judgment scores) |
| ge/dev_class | 8400-8419 | 20 | 11 | graph-endpoint development screens |
| ge/dev_class_ideal | 8400-8419 | 20 | 5 | graph-endpoint development screens |
| ge/dev_class_laplr | 8400-8419 | 20 | 2 | graph-endpoint development screens |
| ge/dev_cut | 8400-8419 | 20 | 4 | graph-endpoint development screens |
| ge/dev_dose | 8400-8419 | 20 | 10 | graph-endpoint development screens |
| ge/dev_harsh | 8400-8419 | 20 | 6 | graph-endpoint development screens |
| ge/dev_k | 8400-8419 | 20 | 4 | graph-endpoint development screens |
| ge/dev_lam | 8400-8419 | 20 | 6 | graph-endpoint development screens |
| ge/dev_lamclass | 8400-8419 | 20 | 0 | graph-endpoint development screens |
| ge/dev_late | 8400-8419 | 20 | 4 | graph-endpoint development screens |
| ge/dev_nodes | 8400-8419 | 20 | 6 | graph-endpoint development screens |
| ge/dev_ridge | 8400-8419 | 20 | 5 | graph-endpoint development screens |
| ge/dev_target | 8400-8419 | 20 | 6 | graph-endpoint development screens |
| ge/dev_temp3 | 8400-8419 | 20 | 4 | graph-endpoint development screens |
| ge/dev_traj | 8400-8419 | 20 | 4 | graph-endpoint development screens |
| ge/dev_trans | 8400-8419 | 20 | 3 | graph-endpoint development screens |
| gl/cold | 7300-7334 | 35 | 12 | GLC cold-start confirmation |
| gl/confirm1 | 7100-7134 | 35 | 12 | GL1 confirmation (PREREGISTRATION_G1) |
| gl/confirm2 | 7200-7234 | 35 | 12 | GL2 independent replication |
| gl/dev1 | 7000-7019 | 20 | 24 | graph-learner development screens |
| gl/dev2 | 7000-7019 | 20 | 13 | graph-learner development screens |
| gl/dev3 | 7000-7019 | 20 | 12 | graph-learner development screens |
| gl/dev_cold | 7000-7019 | 20 | 6 | graph-learner development screens |
| gm/confirm1 | 8100-8134 | 35 | 17 | GM1 / GM2 confirmations (noise dose, RT13-off candidate) |
| gm/confirm2 | 8150-8184 | 35 | 17 | GM1 / GM2 confirmations (noise dose, RT13-off candidate) |
| gm/dev1 | 8000-8019 | 20 | 12 | graph-mechanism development rounds (about 60 variants) |
| gm/dev10 | 8000-8019 | 20 | 5 | graph-mechanism development rounds (about 60 variants) |
| gm/dev11 | 8000-8019 | 20 | 3 | graph-mechanism development rounds (about 60 variants) |
| gm/dev12 | 8000-8019 | 20 | 3 | graph-mechanism development rounds (about 60 variants) |
| gm/dev13 | 8000-8019 | 20 | 2 | graph-mechanism development rounds (about 60 variants) |
| gm/dev14 | 8000-8019 | 20 | 2 | graph-mechanism development rounds (about 60 variants) |
| gm/dev15 | 8000-8019 | 20 | 8 | graph-mechanism development rounds (about 60 variants) |
| gm/dev16 | 8000-8019 | 20 | 4 | graph-mechanism development rounds (about 60 variants) |
| gm/dev17 | 8000-8019 | 20 | 2 | graph-mechanism development rounds (about 60 variants) |
| gm/dev2 | 8000-8019 | 20 | 11 | graph-mechanism development rounds (about 60 variants) |
| gm/dev3 | 8000-8019 | 20 | 12 | graph-mechanism development rounds (about 60 variants) |
| gm/dev4 | 8000-8019 | 20 | 9 | graph-mechanism development rounds (about 60 variants) |
| gm/dev5 | 8000-8019 | 20 | 4 | graph-mechanism development rounds (about 60 variants) |
| gm/dev6 | 8000-8019 | 20 | 3 | graph-mechanism development rounds (about 60 variants) |
| gm/dev7 | 8000-8019 | 20 | 4 | graph-mechanism development rounds (about 60 variants) |
| gm/dev7_INVALID_cache_key_collision | 8000-8019 | 20 | 4 | INVALID development round (graph cache key collision); kept for the record, never used |
| gm/dev8 | 8000-8019 | 20 | 3 | graph-mechanism development rounds (about 60 variants) |
| gm/dev9 | 8000-8019 | 20 | 5 | graph-mechanism development rounds (about 60 variants) |
| gm/dev_cold | 8000-8019 | 20 | 3 | graph-mechanism development rounds (about 60 variants) |
| gm/quality | 8000-8184 | 90 | 0 | per-seed graph statistics (same-class share, degree) used by the dose-response tests |
| rt/a1 | 8800-8819 | 20 | 0 | red team: per-judgment scores of 4 learners (novelty columns) |
| rt/a1b | 8800-8819 | 20 | 0 | red team: distance of each held-out judgment to the nearest typed anchor |
| rt/a1c | 8800-8819 | 20 | 0 | red team: per-judgment scores baseline / aw8 / aw8_all (anchor-weight check) |
| rt/a2 | 8800-8803 | 4 | 3 |  |
| rt/a3 | 8800-8819 | 20 | 5 | red team: time-blocked evaluation |
| rt/a9 | 8800-8803 | 4 | 5 |  |
| rt/c_a1 | 8850-8884 | 35 | 0 | red team: pre-registered confirmation (RT1) |
| rt/c_a1b | 8850-8884 | 35 | 0 | red team: pre-registered confirmation (RT1) |
| rt/c_a3 | 8850-8853 | 4 | 3 | red team: pre-registered confirmation (RT1) |
| screen1 | 2000-2019 | 20 | 7 | new-directions development screen (seeds 2000-2019) |
| screen10 | 2000-2011 | 12 | 6 | new-directions development screen (seeds 2000-2019) |
| screen11_mix | 2000-2019 | 20 | 5 | new-directions development screen (seeds 2000-2019) |
| screen12_ce | 2000-2019 | 20 | 7 | new-directions development screen (seeds 2000-2019) |
| screen13_capacity | 2000-2011 | 12 | 6 | new-directions development screen (seeds 2000-2019) |
| screen14_gp | 2000-2009 | 10 | 3 | new-directions development screen (seeds 2000-2019) |
| screen2a | 2000-2019 | 20 | 3 | new-directions development screen (seeds 2000-2019) |
| screen2b | 2000-2009 | 10 | 5 | new-directions development screen (seeds 2000-2019) |
| screen3_cal | 2000-2019 | 20 | 4 | new-directions development screen (seeds 2000-2019) |
| screen4 | 2000-2003 | 4 | 3 | new-directions development screen (seeds 2000-2019) |
| screen5 | 2000-2010 | 11 | 5 | new-directions development screen (seeds 2000-2019) |
| screen6 | 2000-2015 | 16 | 3 | new-directions development screen (seeds 2000-2019) |
| screen7_anchor | 2000-2019 | 20 | 9 | new-directions development screen (seeds 2000-2019) |
| screen8_all | 2000-2019 | 20 | 5 | new-directions development screen (seeds 2000-2019) |
| screen9 | 2000-2019 | 20 | 8 | new-directions development screen (seeds 2000-2019) |
| sel1 | 2000-2019 | 20 | 1 | new-directions development screen (seeds 2000-2019) |
| sel2 | 2000-2019 | 20 | 1 | new-directions development screen (seeds 2000-2019) |
| sel5_cold | 2000-2019 | 20 | 3 | new-directions development screen (seeds 2000-2019) |
| sel6 | 2000-2019 | 20 | 2 | new-directions development screen (seeds 2000-2019) |
| sel7_family | 2000-2009 | 10 | 2 | new-directions development screen (seeds 2000-2019) |
| sh/c1_learn | 8300-8334 | 35 | 5 | SH1 learner family (aw8_all, aw8_all_st2lr0.5, lap3_k10, lap3_k10_st4, shuffled control), random selection |
| sh/c1_sel | 8300-8334 | 35 | 1 | SH selection family under lap3_k10 (7 selectors incl. controls) |
| sh/c1_sel4 | 8300-8334 | 35 | 1 | SH selection family under lap3_k10_st4 |
| sh/c2_learn | 8350-8384 | 35 | 5 | SH2 learner family, independent replication |
| sh/c2_sel | 8350-8384 | 35 | 1 | SH selection family under lap3_k10 (7 selectors incl. controls) |
| sh/cold_learn | 8385-8399 | 15 | 5 | SHC learner family, cold start (15 seeds) |
| sh/sdev1 | 8200-8219 | 20 | 2 | selection-htr development screens |
| sh/sdev10 | 8200-8219 | 20 | 2 | selection-htr development screens |
| sh/sdev2 | 8200-8219 | 20 | 2 | selection-htr development screens |
| sh/sdev3 | 8200-8219 | 20 | 5 | selection-htr development screens |
| sh/sdev4 | 8200-8219 | 20 | 3 | selection-htr development screens |
| sh/sdev5 | 8200-8219 | 20 | 6 | selection-htr development screens |
| sh/sdev6 | 8200-8211 | 12 | 9 | selection-htr development screens |
| sh/sdev9 | 8200-8219 | 20 | 2 | selection-htr development screens |

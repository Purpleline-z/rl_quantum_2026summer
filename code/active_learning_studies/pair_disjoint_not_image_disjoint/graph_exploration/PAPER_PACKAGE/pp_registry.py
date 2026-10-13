"""Registry of studies, seed blocks and result directories (the single place where branch names, commits and seed ranges are written down; the README, the provenance table and the
claims table are generated from it).  Commits are the branch tips at import time (2026-10-13) and the commits that ADDED the pre-registration file; tables/prereg_order.csv verifies from git
history that every pre-registration was committed before the first confirmation result file of its study."""

# result directories are relative to code/active_learning_studies/new_directions/results/
STUDIES = {
    "ND": dict(name="new directions (anchor weight, selection x learner)", branch="claude/new-directions-overnight", tip="83e1cb866",
               docs=["new_directions/NEW_DIRECTIONS_RESULTS.md", "new_directions/SUMMARY_ZH.md"],
               prereg=[("PREREGISTRATION_1.md", "b1669ca9d", "new_directions/PREREGISTRATION_1.md", ["confirm1"]), ("PREREGISTRATION_3.md", "1b38f7ecf", "new_directions/PREREGISTRATION_3.md", ["confirm3_cold"]),
                       ("PREREGISTRATION_4.md", "9e8da7f84", "new_directions/PREREGISTRATION_4.md", ["confirm4_htr"]), ("PREREGISTRATION_5.md", "12fff894a", "new_directions/PREREGISTRATION_5.md", ["confirm5_gp"])],
               dev=("2000-2019", ["screen1", "screen2a", "screen2b", "screen3_cal", "screen4", "screen5", "screen6", "screen7_anchor", "screen8_all", "screen9", "screen10", "screen11_mix", "screen12_ce", "screen13_capacity", "screen14_gp", "sel1", "sel2", "sel5_cold", "sel6", "sel7_family"])),
    "GL": dict(name="graph learner (Laplacian regulariser on aw8_all)", branch="claude/graph-learner-20261012", tip="54c742290",
               docs=["graph_exploration/RESULTS_graph_learner.md", "graph_exploration/SUMMARY_ZH_graph_learner.md"],
               prereg=[("PREREGISTRATION_G1.md", "3999406bf", "graph_exploration/PREREGISTRATION_G1.md", ["gl/confirm1", "gl/confirm2", "gl/cold"])],
               dev=("7000-7019", ["gl/dev1", "gl/dev2", "gl/dev3", "gl/dev_cold"])),
    "SH": dict(name="selection x Laplacian learner x HTR (lap3_k10_st4)", branch="claude/selection-htr-20261013", tip="c5074019b",
               docs=["graph_exploration/RESULTS_selection_htr.md", "graph_exploration/SUMMARY_ZH_selection_htr.md"],
               prereg=[("PREREGISTRATION_SH1.md", "232d41d11", "graph_exploration/PREREGISTRATION_SH1.md", ["sh/c1_learn", "sh/c1_sel", "sh/c2_learn", "sh/c2_sel", "sh/c1_sel4"]),
                       ("PREREGISTRATION_SH2_cold_start.md", "1f2623ef8", "graph_exploration/PREREGISTRATION_SH2_cold_start.md", ["sh/cold_learn"])],
               dev=("8200-8219", ["sh/sdev1", "sh/sdev2", "sh/sdev3", "sh/sdev4", "sh/sdev5", "sh/sdev6", "sh/sdev9", "sh/sdev10"])),
    "GM": dict(name="graph mechanism (graph quality, RT13 off)", branch="claude/graph-mechanism-20261013", tip="facb44765",
               docs=["graph_exploration/RESULTS_graph_mechanism.md", "graph_exploration/SUMMARY_ZH_graph_mechanism.md"],
               prereg=[("PREREGISTRATION_GM1.md", "0f33c6f65", "graph_exploration/PREREGISTRATION_GM1.md", ["gm/confirm1", "gm/confirm2"])],
               dev=("8000-8019", [f"gm/dev{i}" for i in range(1, 18) if i != 7] + ["gm/dev_cold"])),
    "GE": dict(name="graph endpoint (classification, unlabelled dose, harsh splits)", branch="claude/graph-endpoint-1013", tip="ab06e78ab",
               docs=["graph_exploration/RESULTS_graph_endpoint.md", "graph_exploration/SUMMARY_ZH_graph_endpoint.md"],
               prereg=[("PREREGISTRATION_GE1.md", "1667da4e1", "graph_exploration/PREREGISTRATION_GE1.md", ["ge/conf1_class", "ge/conf1_dose", "ge/conf1_harsh", "ge/conf2_class", "ge/conf2_dose", "ge/conf2_harsh"])],
               dev=("8400-8419", ["ge/dev_class", "ge/dev_dose", "ge/dev_harsh", "ge/dev_k", "ge/dev_lam", "ge/dev_late", "ge/dev_nodes", "ge/dev_ridge", "ge/dev_target", "ge/dev_temp3", "ge/dev_trans"])),
    "RT": dict(name="red team (anchor proximity, novelty, de-duplication, time blocks, re-implementation)", branch="claude/redteam-20261013", tip="7bc38b214",
               docs=["graph_exploration/PREREGISTRATION_RT1.md", "graph_exploration/results_narrative_redteam.md"],
               prereg=[("PREREGISTRATION_RT1.md", "1ec724ee8", "graph_exploration/PREREGISTRATION_RT1.md", ["rt/c_a1", "rt/c_a1b", "rt/c_a2", "rt/c_a3"])],
               dev=("8800-8819", ["rt/a1", "rt/a1b", "rt/a1c", "rt/a3"])),
}

# seed blocks (inclusive ranges): role -> (study, first, last).  Disjointness is asserted by pp_seedblocks.py.
SEED_BLOCKS = [
    ("ND", "development", 2000, 2019), ("ND", "confirmation (ND1: anchor weight / selection)", 3000, 3034), ("ND", "confirmation (ND3: cold start)", 4000, 4034), ("ND", "confirmation (ND4: HTR)", 5000, 5034),
    ("ND", "confirmation (ND5: GP learner)", 6000, 6034),
    ("GL", "development", 7000, 7019), ("GL", "confirmation 1", 7100, 7134), ("GL", "confirmation 2", 7200, 7234), ("GL", "cold-start confirmation", 7300, 7334),
    ("GM", "development", 8000, 8019), ("GM", "confirmation 1", 8100, 8134), ("GM", "confirmation 2", 8150, 8184),
    ("SH", "development", 8200, 8219), ("SH", "confirmation 1", 8300, 8334), ("SH", "confirmation 2", 8350, 8384), ("SH", "cold-start confirmation", 8385, 8399),
    ("GE", "development", 8400, 8419), ("GE", "confirmation 1", 8500, 8534), ("GE", "confirmation 2", 8550, 8584),
    ("RT", "development (audits)", 8800, 8819), ("RT", "confirmation (RT1)", 8850, 8884),
]
# earlier work (not regenerated by this package): seeds 42,79,123,202,303, 400-459, 500-534, 600-629, 700-1739 (judgment-unit / graph selection studies)

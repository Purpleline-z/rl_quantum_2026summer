# PAPER_PACKAGE — results package for the learner / graph / selection line

Everything an advisor or a paper draft needs, regenerated from the **stored result files** (nothing is re-run, `results/` is never modified): a claims table, six figures, the assumptions-and-limits section, an English abstract-level summary and a one-page Chinese summary, with every number in the prose checked against the generated tables.

```
cd code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/PAPER_PACKAGE
pip install -r requirements.txt        # numpy pandas scipy scikit-learn matplotlib (no torch)
make all                               # tables -> figures -> checked documents -> tests (a few minutes, one core)
```

## What `make all` produces
| script | reads | writes (in `tables/`, `figures/`, delivered documents) |
|---|---|---|
| `pp_seedblocks.py` | all result directories | `inventory.md`, `directory_map.md`, `seed_blocks.md` (asserts disjoint seed blocks and that confirmation directories only contain their block's seeds) |
| `pp_provenance.py` | git history | `prereg_order.md` (pre-registration committed before the first result file of its study), `provenance.md` |
| `pp_confirmations.py` | confirmation directories + each study's frozen spec files and own test code | `confirmations.csv` (every pre-registered test), `verdicts.md` (every success rule) |
| `pp_curves.py` | `confirm1`, `confirm4_htr`, `sh/c1_learn`, `sh/c2_learn` | `curves*.csv` (Fig. 1, 2) |
| `pp_ladder.py` | confirmation directories | `ladder.md/csv` (Fig. 3) |
| `pp_selection.py` | selection directories | `selection.md/csv` (Fig. 4) |
| `pp_dose.py` | `gm/*`, `gl/*`, `ge/*` | `dose_*.csv` (Fig. 5) |
| `pp_strata.py` | `rt/a1, a1b, a1c` (+ `rt/c_a1, c_a1b` once complete) | `strata.csv`, `strata_summary.md`, `anchor_fractions.csv`, `novelty.csv` (Fig. 6) |
| `pp_redteam_extra.py` | `rt/a3` (+ `rt/a2`) | `redteam_extra.md` |
| `pp_wcurse.py`, `pp_limits.py` | development + confirmation directories, `results/pair_graph_summary.csv`, `rt/a6.md` | `winners_curse.md`, `limits_facts.csv` |
| `pp_claims.py` | all tables above | `claims_table.md` (one page), `claims_appendix.md`, `claims.csv` |
| `figures/fig*.py` (one per figure) | tables | `figures/*.png`, `*.pdf` |
| `check_prose.py` + `build_docs.py` | `prose/*.src.md` | `ABSTRACT_EN.md`, `ADVISOR_SUMMARY_ZH.md`, and in `graph_exploration/`: `RESULTS_paper_package.md`, `SUMMARY_ZH_paper_package.md`, `HANDOFF_paper_package.md`, `GPU_TASKS_paper_package.md` |

Edit the prose in `prose/*.src.md` (every data number is followed by a marker such as `+0.021⟦cl.L5.A.auc.gain⟧`; see the docstring of `check_prose.py`), never the delivered files. `make docs` fails if a prose number does not match its table.

## Conventions
Gain = mean over seeds of the per-seed mean (over budgets 10/20/40/60 and 3 random label draws) of the paired difference on identical label sets; for log-loss the sign is flipped (positive = better). Intervals = 95%⟦-⟧ bootstrap over seeds. Tests = two-sided Wilcoxon over seeds, Holm within the family frozen in the pre-registration. Splits A / B are the two held-out designs. **Same 168 labelled groups re-split in every seed**: p-values are optimistic. "Anchor-far" = both images of a held-out judgment farther than 0.03⟦-⟧ (1 − cosine) from every typed ideal image.

## Result directories (`code/active_learning_studies/new_directions/results/`)
{{INCLUDE:tables/directory_map.md}}

## Seed blocks (all disjoint; earlier work used 42, 79, 123, 202, 303, 400–459, 500–534, 600–629, 700–1739; 8000–8199, 8400–8799 and 8885–8999 are reserved for parallel conversations, and this package used none: it ran nothing)
{{INCLUDE:tables/seed_blocks.md}}

## Branches (tip commits at import time; commits of the pre-registration files are in `tables/prereg_order.md`)
{{INCLUDE:tables/provenance.md}}

The result trees of `claude/graph-mechanism-20261013` and `claude/graph-endpoint-1013` (and the red-team, selection-htr, graph-learner and new-directions trees already contained in the base branch) are copied in unchanged; `sync_branches.sh` refreshes them from the branches (the red team's confirmation block 8850–8884 is still being written there; `pp_strata.py` includes it automatically once all 70⟦-⟧ files of `rt/c_a1` and `rt/c_a1b` exist; `pp_registry.py` has the tip commits).

## Not included (yet)
Branches that had no result files when this package was built: `claude/representation-20261013`, `claude/anchor-exchange-1013`, `claude/temporal-graph-20261013`; the red team's confirmation block; per-judgment (anchor-far) versions of the standard confirmations (not stored; see `GPU_TASKS_paper_package.md`). The earlier line of work (judgment-unit protocol, selection under the old head, graph selection rules) appears only as *documented-only* rows in the claims appendix.

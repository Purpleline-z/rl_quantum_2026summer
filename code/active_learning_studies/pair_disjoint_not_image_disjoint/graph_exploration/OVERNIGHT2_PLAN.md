# Overnight run 2 (started 2026-10-10): graph rules under the re-tuned head schedule, then an improved graph method

User instruction: re-run the frozen graph rules with the gentler (re-tuned) head schedule; then propose and run next steps for at least 8 hours; the user hopes to see an improved graph method that beats the Random baseline in the morning. Protocol stays fixed (judgment-unit protocol, splits A/B, single-shot + sequential, budgets 10/20/40/60, Random = mean of 5 draws, held-out preference endpoint); the head schedule is now the re-tuned per-budget schedule `results/pair_endpoint_study/schedule_judgment_unit.json` (lr 0.001 for the initial rows and 10/20/40 judgments, lr 0.003 at 60; 100 steps), exactly as in report §5.14-5.15. Honesty rules as before: development seeds are used for design only, confirmation seeds once, no tuning on confirmation seeds, failures reported, all variants tried listed.
Runner: `new_methods_run.sh SPLIT SEEDS ONLY OUTDIR` (cells in `results/new_methods/<OUTDIR>/<SPLIT>/{single,sequential}`), with `JU_GRAPH=1` to expose the graph rules; `graph_exploration/scripts/phase_run.sh PHASE` relaunches a phase (idempotent).

## Phase 1 (P1): frozen graph rules under the re-tuned schedule — fresh seeds 1500-1529, both splits
Rules (frozen, unchanged code): candidates `typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`; controls of the plain rule `typed_decisive_coverage_typeonly`, `typed_decisive_coverage_shuffled`; references `vopt_u` (what works under this schedule, §5.15) and `core_set_relation`; `random`.
H1: each candidate has a positive log-loss gain over Random (seed-level Wilcoxon, Holm over the 3 candidates within each of the 4 blocks); AUC and accuracy reported. H2: plain rule minus its controls (paired, raw p, 2 controls x 4 blocks, not corrected). H3 (descriptive): rule minus `vopt_u`.
Out: `results/new_methods/graph_p1/{A,B}`; analysis `aggregate_overnight2.py p1`.

## Phase 2 (P2): graph-regularised variance-reduction design — development seeds 600-624, both splits (design and choice only)
Idea (literature: Laplacian-regularised optimal design, variance minimisation on graphs; see notes): the best-performing family under the re-tuned schedule is the pool-wide I-optimal design `vopt_u` on the last layer of the own-type head (§5.15). Graph-aware variants change what is minimised or in which space:
- `gvopt_lap`: Laplacian-regularised prior, precision = ridge I + lam * H^T L H / scale (H = hidden features of the images of the graph, L = Laplacian of the kNN graph, lam = 1);
- `gvopt_type`: the pool judgments' weights in the variance sum are the type-posterior relevance (answerability) of the judgment, from label spreading of the typed references;
- `gvopt_prop`: the design is computed on graph-propagated hidden features (one propagation step).
Dev comparison: each variant against Random and against `vopt_u` on seeds 600-624 (F's development seeds). At most ONE graph method (or one combination) is frozen for P3, chosen on dev seeds by mean gain over `vopt_u` and over Random across the four blocks; ties go to the simpler one. Write the frozen choice into the log BEFORE P3.

## Phase 3 (P3): confirmation of the frozen graph method — fresh seeds 1400-1434, both splits
Frozen graph method G, its shuffled-graph control G_shuf, `vopt_u`, `random`. Success claim C1 (user's request): G beats Random — Holm p < 0.05 for log-loss in >= 3 of 4 blocks with non-negative AUC and accuracy gains. Claim C2 (graph helps): G minus G_shuf and G minus `vopt_u` paired differences, reported with raw p; C2 is claimed only if both are positive in a majority of blocks with raw p < 0.05 in at least two.

## Literature reading in the foreground while jobs run
Notes in `graph_exploration/notes_overnight2_literature.md`: Laplacian-regularised optimal design (He et al.), manifold-adaptive experimental design (Cai & He), variance minimisation on graphs (Ji & Han), active learning with Gaussian fields and harmonic functions (Zhu et al.), graph signal sampling, plus what is known about gentle/low-step fine-tuning of linear heads.

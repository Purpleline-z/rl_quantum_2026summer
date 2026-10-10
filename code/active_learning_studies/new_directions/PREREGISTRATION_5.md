# Pre-registration 5 (GP preference learner with the anchor weight)

Written before the confirmatory cells exist. Development (seeds 2000-2009, `results/screen14_gp`, 10 seeds): the Gaussian-process preference learner of `nd_gp.py` (RBF kernel on L2-normalised SimCLR features, one latent function per type over all 1278 images, MAP in the dual, grouped-CV penalty, reference-anchor loss weight 8, trained on the training references only, length-scale fixed at 0.5 x the median distance) improved the held-out preference prediction over the aw8 MLP head: split A AUC +0.018 [+0.002, +0.037], accuracy +0.031 [+0.011, +0.053], log-loss +0.050 [-0.005, +0.110]; split B AUC +0.012 [+0.002, +0.022], accuracy +0.009, log-loss +0.013 (nominal p 0.03-0.5; development only, small n). With the default anchor weight the GP was not better than the MLP head (screen2b), so the comparison here is with weight 8 on both sides.

Confirmatory seeds **6000-6034** (fresh). Random label sets only (3 draws per seed and budget), splits A and B, budgets 10 / 20 / 40 / 60 judgments, initial 10 groups, as in pre-registration 1. Cells: random selection x learner in {baseline, aw8, gp_aw8}.

Families (spec `prereg5_spec.json`, code `nd_confirm.py`; two-sided Wilcoxon over 35 seeds, Holm within a family): **K1 (6 tests)** (random, gp_aw8) vs (random, aw8), AUC / accuracy / log-loss, splits A and B; **K2 (6 tests)** (random, gp_aw8) vs (random, baseline), same metrics.

Decision rule: the GP learner is called better than the anchor-weighted MLP head in a split only if K1 gives Holm < 0.05 for AUC and accuracy in that split. Limits as before (same 168 groups, small held-out sets).

# Pre-registration 2 (ideal-image type accuracy under anchor weighting)

Written before the confirmatory cells exist. Development seeds 2000-2019 (`results/type_accuracy_dev.json`) showed that raising the reference-anchor weight from 0.25 to 2 / 8 / 32 raises the paper's own endpoint, the type accuracy of the 28 outer-test ideal images (win-rate against the references, `frozen.evaluate_model`), from about 0.81-0.835 to 0.85-0.87 (dev gains +0.032 / +0.042 / +0.044).

Confirmatory seeds **3000-3034**; split-B context; learners trained on the references only (no ideal image of the outer test is used); budgets 0 (initial judgments only), 10, 20, 40, 60; 3 random draws. Per seed: mean over budgets and draws of type accuracy(aw) - type accuracy(default 0.25). Tests: two-sided Wilcoxon signed-rank over the 35 seeds for aw in {2, 8, 32}; Holm over these 3 tests; paired bootstrap 95% interval. Code: `nd_type_accuracy.py`, `nd_confirm_type.py`. Everything else (per budget, per class) is exploratory.

Limits: same 168 pair groups and the same 144 ideal images in every seed (seeds re-split the ideal images 60/20/20, so the 28-image outer test overlaps across seeds); a gain here says nothing about real mixed-phase images.

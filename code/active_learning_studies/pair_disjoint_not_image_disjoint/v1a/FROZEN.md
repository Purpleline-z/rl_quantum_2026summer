# V1a frozen setting (committed before the main experiment was run)

Chosen from `results/tuning_grid.csv` (3 x 3 grid, seeds 42/79/123, label sets = initial + {0,10,20,40,60} random pool judgments, 15 fits per setting, criterion = mean **validation** ideal-image accuracy; the 28 test ideal images were not evaluated in tuning):

**lr = 0.01, steps = 300** (mean validation accuracy 0.8595; runner-ups: lr 0.003/300 and lr 0.001/1000 at 0.8548). One setting for every budget and every rule.

Other fixed choices: AdamW, weight decay 1e-4, gradient-norm clip 1.0, full batch, head initialisation = the experiment seed's own (identical at every retrain; ensemble members re-initialise with head seeds 1000+k as in `judgment_unit_study.choose`), dropout inert during training (repository behaviour of `frozen_encoder_reward_head.fit_head`).
Mix weights 0.60 / 0.25 / 0.15; ideal term 0.7 trajectory + 0.3 other-type ideal; T = 1087 trajectory images for seed 42 (1124 minus 37 byte-identical to an ideal image; checked per seed in the output CSV column `T`).
Differences between the grid cells are of the order of the noise of 15 fits x 28 images (about 0.01-0.02), so the choice is a weak preference, not a sharp optimum.

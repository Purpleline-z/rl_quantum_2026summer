### Frozen-feature baselines, paper split (5 splits, 28 test images each)

Reference row for the paired difference: `simclr_resnet18` + `1nn` (same split). 'better/worse' count splits.

| extractor | head | splits | accuracy | macro-F1 | HTR recall | HTR precision | Δacc vs ref | better / worse |
|---|---|---|---|---|---|---|---|---|
| random_resnet18_init0 | logreg | 5 | 0.907 ± 0.048 | 0.898 ± 0.047 | 0.80 ± 0.14 | 0.90 ± 0.15 | +0.036 | 3 / 1 |
| random_resnet18_init0 | 1nn | 5 | 0.900 ± 0.039 | 0.883 ± 0.050 | 0.68 ± 0.18 | 0.92 ± 0.11 | +0.029 | 3 / 2 |
| random_resnet18_init2 | 1nn | 5 | 0.886 ± 0.064 | 0.870 ± 0.078 | 0.72 ± 0.23 | 0.85 ± 0.14 | +0.014 | 3 / 2 |
| raw_pixels_64 | logreg | 5 | 0.886 ± 0.030 | 0.879 ± 0.032 | 0.88 ± 0.18 | 0.88 ± 0.12 | +0.014 | 3 / 2 |
| random_resnet18_init1 | logreg | 5 | 0.886 ± 0.016 | 0.872 ± 0.018 | 0.76 ± 0.17 | 0.86 ± 0.13 | +0.014 | 3 / 1 |
| random_resnet18_init1 | 1nn | 5 | 0.886 ± 0.047 | 0.872 ± 0.056 | 0.72 ± 0.23 | 0.92 ± 0.11 | +0.014 | 3 / 2 |
| raw_pixels_64 | 1nn | 5 | 0.879 ± 0.054 | 0.865 ± 0.066 | 0.72 ± 0.23 | 0.92 ± 0.11 | +0.007 | 3 / 2 |
| simclr_resnet18 | 1nn | 5 | 0.871 ± 0.041 | 0.856 ± 0.046 | 0.76 ± 0.09 | 0.80 ± 0.18 | +0.000 | 0 / 0 |
| peak_profile_24 | 1nn | 5 | 0.871 ± 0.041 | 0.843 ± 0.050 | 0.60 ± 0.20 | 0.85 ± 0.20 | +0.000 | 1 / 2 |
| peak_profile_24 | logreg | 5 | 0.864 ± 0.085 | 0.850 ± 0.099 | 0.68 ± 0.27 | 0.96 ± 0.09 | -0.007 | 2 / 2 |
| peak_profile_24 | 5nn | 5 | 0.857 ± 0.025 | 0.833 ± 0.029 | 0.60 ± 0.14 | 0.74 ± 0.16 | -0.014 | 0 / 2 |
| random_resnet18_init2 | logreg | 5 | 0.857 ± 0.051 | 0.835 ± 0.065 | 0.64 ± 0.22 | 0.88 ± 0.11 | -0.014 | 3 / 2 |
| simclr_resnet18 | 5nn | 5 | 0.857 ± 0.036 | 0.836 ± 0.047 | 0.72 ± 0.18 | 0.70 ± 0.19 | -0.014 | 1 / 3 |
| random_resnet18_init1 | 5nn | 5 | 0.850 ± 0.053 | 0.823 ± 0.066 | 0.56 ± 0.22 | 0.87 ± 0.18 | -0.021 | 1 / 3 |
| simclr_resnet18 | logreg | 5 | 0.850 ± 0.047 | 0.827 ± 0.064 | 0.64 ± 0.22 | 0.69 ± 0.21 | -0.021 | 2 / 3 |
| random_resnet18_init0 | 5nn | 5 | 0.843 ± 0.020 | 0.812 ± 0.030 | 0.52 ± 0.18 | 0.82 ± 0.17 | -0.029 | 1 / 3 |
| random_resnet18_init2 | 5nn | 5 | 0.836 ± 0.041 | 0.806 ± 0.052 | 0.52 ± 0.18 | 0.87 ± 0.18 | -0.036 | 1 / 3 |
| raw_pixels_64 | 5nn | 5 | 0.814 ± 0.030 | 0.783 ± 0.041 | 0.52 ± 0.18 | 0.75 ± 0.23 | -0.057 | 0 / 3 |

Not run (weights could not be downloaded in this environment): imagenet_resnet18, imagenet_resnet18_paper_norm, imagenet_resnet50, imagenet_vit_b_16

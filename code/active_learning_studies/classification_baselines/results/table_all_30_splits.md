### Frozen-feature controls, 5 paper splits + 25 extra splits (30 splits, 28 test images each)

Reference row for the paired difference: `simclr_resnet18` + `1nn` (same split). 'better/worse' count splits.

| extractor | head | splits | accuracy | macro-F1 | HTR recall | HTR precision | Δacc vs ref | better / worse |
|---|---|---|---|---|---|---|---|---|
| raw_pixels_64 | logreg | 30 | 0.914 ± 0.045 | 0.909 ± 0.048 | 0.90 ± 0.11 | 0.90 ± 0.13 | +0.037 | 20 / 7 |
| random_resnet18_init1 | logreg | 30 | 0.906 ± 0.044 | 0.895 ± 0.056 | 0.84 ± 0.17 | 0.85 ± 0.15 | +0.029 | 17 / 5 |
| random_resnet18_init0 | 1nn | 30 | 0.905 ± 0.044 | 0.892 ± 0.053 | 0.78 ± 0.16 | 0.89 ± 0.12 | +0.027 | 19 / 7 |
| raw_pixels_64 | 1nn | 30 | 0.902 ± 0.053 | 0.892 ± 0.061 | 0.81 ± 0.16 | 0.90 ± 0.12 | +0.025 | 19 / 4 |
| random_resnet18_init2 | 1nn | 30 | 0.900 ± 0.054 | 0.888 ± 0.062 | 0.79 ± 0.15 | 0.89 ± 0.13 | +0.023 | 18 / 5 |
| random_resnet18_init0 | logreg | 30 | 0.900 ± 0.038 | 0.889 ± 0.046 | 0.83 ± 0.17 | 0.84 ± 0.15 | +0.023 | 15 / 5 |
| random_resnet18_init2 | logreg | 30 | 0.895 ± 0.050 | 0.880 ± 0.058 | 0.76 ± 0.18 | 0.86 ± 0.14 | +0.018 | 15 / 5 |
| random_resnet18_init1 | 1nn | 30 | 0.894 ± 0.043 | 0.883 ± 0.050 | 0.79 ± 0.16 | 0.89 ± 0.12 | +0.017 | 15 / 4 |
| simclr_resnet18 | 5nn | 30 | 0.886 ± 0.047 | 0.871 ± 0.054 | 0.83 ± 0.18 | 0.76 ± 0.15 | +0.008 | 11 / 10 |
| peak_profile_24 | logreg | 30 | 0.886 ± 0.052 | 0.873 ± 0.059 | 0.74 ± 0.18 | 0.87 ± 0.15 | +0.008 | 13 / 9 |
| simclr_resnet18 | logreg | 30 | 0.879 ± 0.058 | 0.864 ± 0.065 | 0.75 ± 0.15 | 0.79 ± 0.18 | +0.001 | 13 / 9 |
| simclr_resnet18 | 1nn | 30 | 0.877 ± 0.051 | 0.861 ± 0.061 | 0.78 ± 0.16 | 0.78 ± 0.14 | +0.000 | 0 / 0 |
| random_resnet18_init2 | 5nn | 30 | 0.869 ± 0.053 | 0.852 ± 0.064 | 0.73 ± 0.21 | 0.87 ± 0.15 | -0.008 | 9 / 14 |
| peak_profile_24 | 1nn | 30 | 0.868 ± 0.048 | 0.845 ± 0.058 | 0.72 ± 0.18 | 0.81 ± 0.15 | -0.010 | 7 / 11 |
| random_resnet18_init1 | 5nn | 30 | 0.863 ± 0.054 | 0.847 ± 0.063 | 0.78 ± 0.17 | 0.83 ± 0.17 | -0.014 | 7 / 13 |
| random_resnet18_init0 | 5nn | 30 | 0.854 ± 0.052 | 0.833 ± 0.063 | 0.73 ± 0.21 | 0.79 ± 0.18 | -0.024 | 5 / 14 |
| raw_pixels_64 | 5nn | 30 | 0.849 ± 0.051 | 0.829 ± 0.061 | 0.75 ± 0.20 | 0.77 ± 0.17 | -0.029 | 5 / 17 |
| peak_profile_24 | 5nn | 30 | 0.846 ± 0.039 | 0.823 ± 0.048 | 0.70 ± 0.19 | 0.72 ± 0.14 | -0.031 | 3 / 16 |

Not run (weights could not be downloaded in this environment): imagenet_resnet18, imagenet_resnet18_paper_norm, imagenet_resnet50, imagenet_vit_b_16

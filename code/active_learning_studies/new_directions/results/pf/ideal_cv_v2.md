| block | clf | acc | acc_8bit | acc_(1 x 1) | acc_c(6 x 2) | acc_(√13 x √13) | acc_HTR | n_seeds |
|---|---|---|---|---|---|---|---|---|
| SimCLR final feature (as used by all learners) | 1nn | 0.912 | 0.929 | 1.000 | 0.950 | 0.843 | 0.810 | 20 |
| SimCLR, 16-bit ideals decoded correctly | 1nn | 0.873 | 0.925 | 1.000 | 0.925 | 0.821 | 0.660 | 20 |
| physics pfa | 1nn | 0.896 | 0.911 | 0.975 | 0.919 | 0.871 | 0.770 | 20 |
| physics pfb (peak table) | 1nn | 0.921 | 0.958 | 0.975 | 0.969 | 0.850 | 0.860 | 20 |
| physics pfc (all blocks) | 1nn | 0.925 | 0.956 | 0.975 | 0.969 | 0.900 | 0.810 | 20 |
| random projection of SimCLR (dim of pfa) | 1nn | 0.902 | 0.931 | 1.000 | 0.925 | 0.807 | 0.840 | 20 |
| SimCLR + pfa (z-scored concat) | 1nn | 0.916 | 0.944 | 1.000 | 0.950 | 0.850 | 0.820 | 20 |
| SimCLR (16-bit decoded) + pfa | 1nn | 0.902 | 0.939 | 1.000 | 0.950 | 0.836 | 0.760 | 20 |
| SimCLR final feature (as used by all learners) | logreg | 0.923 | 0.952 | 1.000 | 0.956 | 0.886 | 0.800 | 20 |
| SimCLR, 16-bit ideals decoded correctly | logreg | 0.895 | 0.961 | 1.000 | 0.963 | 0.821 | 0.720 | 20 |
| physics pfa | logreg | 0.923 | 0.955 | 0.994 | 0.956 | 0.864 | 0.840 | 20 |
| physics pfb (peak table) | logreg | 0.927 | 0.969 | 1.000 | 0.963 | 0.871 | 0.830 | 20 |
| physics pfc (all blocks) | logreg | 0.945 | 0.969 | 1.000 | 0.969 | 0.893 | 0.890 | 20 |
| random projection of SimCLR (dim of pfa) | logreg | 0.873 | 0.930 | 1.000 | 0.956 | 0.729 | 0.740 | 20 |
| SimCLR + pfa (z-scored concat) | logreg | 0.932 | 0.957 | 1.000 | 0.956 | 0.886 | 0.850 | 20 |
| SimCLR (16-bit decoded) + pfa | logreg | 0.929 | 0.963 | 1.000 | 0.963 | 0.879 | 0.830 | 20 |

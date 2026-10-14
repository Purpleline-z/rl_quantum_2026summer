# HTR label-consistency report (static: labels and frozen SimCLR geometry only)

## 1. What data there is

| type        |   1 |   2 |   not_apply |   tie |
|:------------|----:|----:|------------:|------:|
| (1 x 1)     |  38 |  36 |          28 |    29 |
| (√13 x √13) |  39 |  55 |          21 |    33 |
| HTR         |  26 |  19 |          83 |     1 |
| c(6 x 2)    |  22 |  29 |          52 |    10 |

* (pair, type) judged more than once: **0** (unordered image pairs judged more than once per type: 0) -> no repeated comparisons, no inter-labeller overlap exist; the ceiling cannot come from repeats.
* decisive HTR judgments: **45** (image 1 won 26, image 2 won 19); not_apply 83, tie 1; distinct images in decisive HTR judgments: 83.
* labellers (decisive HTR): {'AJ': 27, 'HL': 13, 'SY': 5}; confidence {'none': 30, 'Confident': 14, 'Somewhat sure': 1}; runs of the two images: {'RR220206A': 48, 'RR220204A': 25, 'RR220411A': 17}; both frames from the same run: 7 of 45.

Comparison kind by nearest ideal-image class of the two frames (decisive judgments of each type; 1-NN on centred SimCLR features, the four active classes):

| cls_pair   |   (1 x 1) |   (√13 x √13) |   HTR |   c(6 x 2) |
|:-----------|----------:|--------------:|------:|-----------:|
| 1x1-1x1    |        14 |            13 |     1 |          2 |
| 1x1-HTR    |        13 |             6 |    13 |          5 |
| 1x1-RT13   |        22 |            34 |     3 |         13 |
| 1x1-c6x2   |         2 |             2 |     1 |          1 |
| HTR-HTR    |         0 |             0 |     1 |          0 |
| HTR-RT13   |        10 |            23 |    25 |         11 |
| HTR-c6x2   |         0 |             1 |     0 |          1 |
| RT13-RT13  |        10 |            11 |     1 |         15 |
| RT13-c6x2  |         3 |             4 |     0 |          3 |

## 2. Do near-identical comparisons agree? (oriented nearest-neighbour label agreement, same type, decisive)

D = max over the two matched images of 1 - cos (SimCLR); the typical distance between a test frame and its nearest training frame is 0.0002-0.02, between different-type ideal images 0.58 (median).

| D bin     |   ('n judgment pairs', '(1 x 1)') |   ('n judgment pairs', '(√13 x √13)') |   ('n judgment pairs', 'HTR') |   ('n judgment pairs', 'c(6 x 2)') |   ('label agreement', '(1 x 1)') |   ('label agreement', '(√13 x √13)') |   ('label agreement', 'HTR') |   ('label agreement', 'c(6 x 2)') |
|:----------|----------------------------------:|--------------------------------------:|------------------------------:|-----------------------------------:|---------------------------------:|-------------------------------------:|-----------------------------:|----------------------------------:|
| 0-0.01    |                                25 |                                    32 |                             8 |                                  5 |                            0.72  |                                1     |                        1     |                             0.8   |
| 0.01-0.03 |                                34 |                                    37 |                            11 |                                 10 |                            0.765 |                                1     |                        1     |                             0.9   |
| 0.03-0.1  |                                94 |                                   157 |                            45 |                                 26 |                            0.734 |                                0.987 |                        1     |                             0.885 |
| 0.1-0.3   |                               392 |                                   817 |                           213 |                                161 |                            0.579 |                                0.969 |                        0.967 |                             0.814 |
| 0.3-9     |                              2156 |                                  3328 |                           713 |                               1073 |                            0.482 |                                0.919 |                        0.655 |                             0.682 |

Leave-one-out 1-NN **label predictor** (predict a judgment's winner from its nearest other same-type comparison, oriented; uses no model, no image-disjointness):

| type        |   n |   1-NN accuracy |   median nn_D |   accuracy if nn_D<0.1 |   n(nn_D<0.1) |
|:------------|----:|----------------:|--------------:|-----------------------:|--------------:|
| (1 x 1)     |  74 |           0.703 |         0.022 |                  0.705 |            61 |
| (√13 x √13) |  94 |           0.968 |         0.027 |                  0.988 |            80 |
| HTR         |  45 |           0.978 |         0.041 |                  1     |            35 |
| c(6 x 2)    |  51 |           0.706 |         0.053 |                  0.771 |            35 |

## 3. Transitivity (directed winner -> loser graph, HTR decisive judgments)

|   merge radius |   nodes |   edges |   2-cycles |   3-cycles |
|---------------:|--------:|--------:|-----------:|-----------:|
|          0     |      83 |      45 |          0 |          0 |
|          0.003 |      42 |      41 |          0 |          0 |
|          0.01  |      25 |      23 |          0 |          0 |
|          0.03  |      16 |      16 |          0 |          0 |
|          0.1   |       0 |       0 |          0 |          0 |

(2-cycle = the same pair of merged image clusters judged in both directions; 3-cycle = A>B, B>C, C>A.)  The graph is almost a matching, so transitivity has little power.

## 4. Model-free rules on the decisive HTR judgments (accuracy; sign +1 = image 1 won)

|                                             |   accuracy |
|:--------------------------------------------|-----------:|
| image 2 listed second wins (always -1)      |      0.422 |
| image 1 wins (always +1)                    |      0.578 |
| later frame wins (same-run pairs only, n=7) |      1     |
| higher temperature wins (all)               |      0.644 |
| lower temperature wins (all)                |      0.356 |
| closer to ideal HTR wins                    |      0.978 |
| farther from ideal RT13 wins                |      0.8   |
| (d_RT13 - d_HTR) larger wins                |      0.8   |

# Image Representation Analysis

## Question

This exploratory study asks whether the shipped SimCLR encoder, ImageNet ResNet-18 features, and raw pixels separate the available ideal-image classes differently.

## Data, split, and model version

The analysis uses the image population recorded in each result manifest: labelled ideal images and unlabeled trajectory images. Metrics that require labels use only the labelled ideal-image subset and their recorded cross-validation folds; trajectory images have no inferred labels. Encoders are the shipped SimCLR ResNet-18 checkpoint and torchvision ImageNet ResNet-18 as recorded in the outputs.

## Use in the technical report

The report may cite the k-nearest-neighbor metrics as representation context. PCA and t-SNE are exploratory figures only and do not establish downstream Bradley–Terry or active-learning performance.

## Comparability limits

This is neither the pair-disjoint active-learning study nor the strict image-disjoint study. It does not use their training splits, budgets, labels, or outer-test endpoint.

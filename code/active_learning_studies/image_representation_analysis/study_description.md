# Image Representation Analysis

## Question

Before interpreting active-learning behavior, this study asks a simpler representation question: do raw pixels, the shipped RHEED SimCLR encoder, and ImageNet ResNet-18 organize the available ideal reconstruction images differently? It measures encoder-space structure without using pairwise preference outcomes or active-selection results.

## Inputs and method

The recorded population contains 1,278 images: 154 labeled ideal reconstruction images and 1,124 unlabeled trajectory images. Raw pixels, frozen SimCLR embeddings, and frozen ImageNet embeddings are extracted from the same image manifest. Five-nearest-neighbor cross-validation uses only labeled ideal images. PCA and t-SNE place both ideal and trajectory images in two dimensions for visual inspection; unlabeled trajectory frames never receive an inferred reconstruction label.

## Observed result

The recorded five-nearest-neighbor accuracies on ideal images are 0.832 for raw pixels, 0.884 for the shipped SimCLR representation, and 0.896 for ImageNet ResNet-18. [Metrics table](results/selected_summaries/representation_exploration/tables/representation_metrics.csv)

## Meaning

Both learned encoders put same-class ideal images closer together than raw pixels do under this local-neighborhood diagnostic. ImageNet's small advantage over the shipped SimCLR checkpoint shows that domain-specific pretraining is not automatically superior for the available ideal-image classes. It does not identify the reason: possible causes include upstream pretraining population, augmentation, checkpoint choice, and downstream fine-tuning protocol.

PCA and t-SNE are useful for seeing overlap, outliers, and trajectory coverage. They do not measure the pairwise Bradley--Terry objective and they do not establish an active-learning advantage. Their practical role is to guide what should be tested downstream: initialization, image preprocessing, and possible out-of-distribution trajectory regions.

## Research decision

Use the representation result as motivation to compare SimCLR and ImageNet under identical downstream partitions and training budgets, rather than selecting an encoder from a visualization. The completed downstream encoder screen and the in-progress budget-aware calibration are the appropriate tests of that decision.

# Ideal-image classification baselines (paper protocol)

Question: how do simple and standard image-classification baselines compare with the reward-model pipeline when everything uses **the same split, the same labelled ideal images and the same outer-test images**?

## Protocol (why it is comparable)

- The split is not re-implemented. `baseline_protocol.load_split(seed)` calls the same `Experiment.load_and_split` as Task 3c and `compute_frozen_encoder_nearest_neighbour_baseline.py` (`exclude_all_ideal_identities_from_pairwise=True`). Per seed, four active types: about **88 reference images, 28 outer-test images, 28 utility-validation images**, de-duplicated by SHA-256 content identity (Twinned(2×1) excluded, as in the paper). `test_split_is_the_papers_and_identity_disjoint` fails if any content identity occurs in two of the three sets.
- Labelled data for a baseline = the **reference images only** (the same absolute labels the Bradley–Terry model sees as anchors). Baselines use no pairwise labels, so they are the "no pair labels" reference points for the acquisition studies.
- The outer test is scored once. Hyper-parameters (logistic-regression `C`, fine-tuning epoch) are chosen on the references (inner cross-validation) or on the utility-validation images, never on the outer test (`test_logreg_depends_only_on_references…`, `test_finetune_epoch_is_selected_on_utility_validation…`).
- Protocol check: frozen SimCLR + 1-NN on this split reproduces, seed by seed, the numbers already in the repository (0.857, 0.857, 0.893, 0.821, 0.929; mean 0.871 ± 0.041) — `test_simclr_1nn_reproduces_the_committed_paper_baseline`.
- **Not used as the primary protocol: cross-validation over all 150 images.** It would train on about 120 labelled images instead of 88 and test on different images, so its numbers could not be compared with the paper's. (Hash-disjoint folds would not leak test images into training, but the numbers would still not be comparable.)
- Seeds: the five paper seeds (42, 79, 123, 202, 303) are the primary result. 25 further random splits of the *same* protocol (seeds 400–424) are a supplementary robustness check on how much the 28-image test set matters; the splits overlap, so the sd is split-to-split variability, not an independent-sample standard error, and no significance tests are attached.

## Baselines run

| name | what it is | needs weights? |
|---|---|---|
| `raw_pixels_64` | 64×64 grayscale pixels, z-scored with reference statistics (logistic regression: PCA to 64 dims fitted on the references) | no |
| `peak_profile_24` | 24 untrained intensity-profile features of the peak-aware paper | no |
| `random_resnet18_init{0,1,2}` | ResNet-18, random weights, three initialisations (lower bound: what does pre-training add?) | no |
| `simclr_resnet18` | laboratory SimCLR ResNet-18, frozen | local checkpoint |
| `finetune_{simclr,random}_resnet18` | supervised fine-tuning, cross-entropy on references only, epoch chosen on utility validation | local checkpoint |
| `imagenet_resnet18[_paper_norm]`, `imagenet_resnet50`, `imagenet_vit_b_16` | torchvision ImageNet models | **yes — not run, see below** |

Heads on frozen features: `1nn` (cosine to the references, identical to the existing baseline), `5nn`, `logreg` (L2, `C` chosen by 5-fold CV inside the references).
Metrics: accuracy, macro-F1, HTR precision / recall (HTR is the type the lab cares most about), per-class recall; predictions per test image are saved for paired analyses.

## Not run: weights that cannot be downloaded here

The environment's network policy rejects `download.pytorch.org` (ImageNet weights), `huggingface.co` (CLIP, many DINOv2 builds) and `dl.fbaipublicfiles.com` (DINOv2 official). The extractors are registered and fail with `WeightsUnavailable`; the runner records them in `unavailable.json` and does **not** substitute another model. DINOv2 and CLIP extractors are not written yet (they cannot be tested here). Once the hosts are allowed, `run_baselines.py --extractors imagenet_resnet18,...` resumes without changing existing rows.

## Reproduce

```bash
cd code/active_learning_studies/classification_baselines
export PYTHONPATH=../../active_learning_program
python3 run_baselines.py --seeds paper --out results/frozen_paper_seeds
python3 run_baselines.py --seeds extra --out results/frozen_extra_seeds
python3 finetune_baseline.py --seeds paper --out results/finetune_paper_seeds
python3 aggregate_baselines.py results/frozen_paper_seeds --out results/table_paper_seeds
python3 -m pytest ../../active_learning_program/code_behavior_tests/test_classification_baselines.py
```
Results: `results/table_paper_seeds.md`, `results/table_all_30_splits.md` (frozen), `results/table_finetune.md`.

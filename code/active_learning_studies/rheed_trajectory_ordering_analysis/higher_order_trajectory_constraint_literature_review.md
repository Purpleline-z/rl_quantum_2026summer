# Higher-Order RHEED Trajectory Constraint: Literature Review and Design Rationale

## Purpose

This study represents an experimentally motivated sequence as a **soft, whole-trajectory constraint**, rather than appending time or process variables as frame-level metadata.  Given external framewise probabilities for six states—`(1 x 1)`, `Bad`, `Twinned(2 x 1)`, `c(6 x 2)`, `(√13 x √13)`, and `HTR`—a Viterbi decoder selects a high-probability path while penalising three deviations:

1. the first decoded state is not `(1 x 1)`;
2. a non-`(1 x 1)`, non-`Bad` reconstruction appears before `Bad`; and
3. a non-HTR reconstruction follows HTR.

The penalties are deliberately soft and are reported at weak (1.25×), moderate (2×), and strong (5×) evidence-equivalent levels.  A framewise score may therefore override the physics preference.  This is essential because the proposed rule is an empirical regularity, not a universal law or a ground-truth label.

## Physical basis and its limits

The immediate system-specific motivation is the laboratory observation that trajectories begin at `(1 x 1)`, typically pass through a disordered `Bad` regime before other reconstruction types, and that HTR appears last.  The oxygen-vacancy explanation is physically plausible: vacancy formation and stability in strained SrTiO3 are temperature- and strain-dependent, as quantified by finite-temperature first-principles work ([Zhou, Chu, and Cazorla, *Scientific Reports*, 2021](https://doi.org/10.1038/s41598-021-91018-4)).  A direct RHEED/LEED account of high-temperature vacuum annealing also reports an HTR-like, broadened reconstruction consistent with a heavily disordered oxygen-vacancy-related surface, while noting that the transition does not occur in every case ([Cornell dissertation, 2024](https://ecommons.cornell.edu/bitstreams/802d9d69-9c8b-4830-9152-115fa7e5e8c6/download)).

The broader reconstruction literature establishes that SrTiO3 surface reconstructions respond strongly to growth stoichiometry and preparation conditions; it should not be read as proving the exact HTR-last sequence in this dataset ([Kajdos and Stemmer, 2014](https://arxiv.org/abs/1410.8830)).  Accordingly, this project treats the three statements as candidate trajectory priors to be sensitivity-tested, not as automatically valid labels.  A newer system-specific experimental sequence with the same trajectory definition was not located in the reviewed sources, so no newer generic oxide paper is substituted as proof of the laboratory rule.

## Why soft constrained decoding

Viterbi decoding is the standard dynamic-programming solution for a most-probable state sequence when local evidence and transition structure are available ([Rabiner, 1989](https://doi.org/10.1109/5.18626)).  More generally, structured-prediction research separates local predictors from declarative constraints, allowing constraints to be audited and adjusted independently ([Chang et al., 2008](https://cogcomp.seas.upenn.edu/papers/CRRR08.pdf)).  This is exactly the present need: the image classifier supplies local evidence, while physics supplies a cross-frame preference.

The implementation uses the smallest necessary memory: whether `Bad` has occurred.  That memory lets the decoder penalise an apparent transition from the initial `(1 x 1)` regime directly into another reconstruction before `Bad`; it does not add an unvalidated transition graph.  `HTR → non-HTR` is penalised, not forbidden.  The raw classifier argmax remains in every output row, so disagreements are visible rather than erased.

## Why not the alternatives now

| Alternative | Why it is not the initial method | Literature basis |
|---|---|---|
| Frame-level metadata embedding | A temperature/time feature changes the per-image predictor but does not directly express “HTR comes last” or “Bad must precede another reconstruction.” It also risks exploiting run identity rather than image evidence. | Declarative constraints are explicitly distinct from local feature design in [Chang et al., 2008](https://cogcomp.seas.upenn.edu/papers/CRRR08.pdf). |
| Hard rule filter | It could force an incorrect sequence whenever the laboratory regularity has an exception. It would hide classifier–physics conflicts. | Soft constraint frameworks retain a tunable trade-off between data fit and constraints, e.g. [Ganchev et al., 2010](https://jmlr.org/papers/v11/ganchev10a.html). |
| Train a CRF/HMM or end-to-end structured neural model | This requires reliably ordered, state-labelled trajectories for estimating emissions/transitions and a held-out trajectory evaluation design. Those labels are not currently available. | Constraint-embedded structured prediction is viable but is a more complex learning system, not a substitute for absent labels ([Jiang et al., 2022](https://jmlr.org/papers/v23/21-1484.html)). |
| Joint neuro-symbolic optimisation | It is valuable once there is enough supervision to learn constraint weights, but it introduces a new trainable model and hyperparameter-selection problem. | [Dragone et al., 2021](https://arxiv.org/abs/2103.17232) show the benefit of coupling neural scores and constraint solving, particularly with limited data; this project first keeps those components independently auditable. |
| Fine-tune the image classifier to obey the rule | It would entangle the unverified prior with visual training and make it harder to determine whether improvement comes from image evidence or sequence forcing. | Constraint-aware decoding can impose structure without fine-tuning the base model ([Geng et al., 2023](https://aclanthology.org/2023.emnlp-main.674/)). |

## Required evidence before a scientific performance claim

The code can be tested on synthetic probabilities now, but it must not be used to claim improved reconstruction accuracy until all of the following exist:

1. an immutable downstream Classifier2 checkpoint or a clearly labelled reproduction checkpoint;
2. a documented calibration from its scores to the six required state probabilities, including `Bad`;
3. ordered trajectories with independent state annotations for evaluation; and
4. a predeclared comparison of raw classifier predictions against all three soft-constraint strengths.

Until then, the deliverable is an auditable implementation and a physics-motivated sensitivity analysis, not a validated temporal classifier.

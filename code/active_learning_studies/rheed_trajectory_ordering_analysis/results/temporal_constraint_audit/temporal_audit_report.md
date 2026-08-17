# Temporal Constraint Audit

This is a descriptive filename/order audit, not a temporal classifier experiment. The physics-team statements remain tentative configuration only; they were not converted into labels, a loss, a decoding rule, or a selection constraint.

## What is available now

- `trajectory_frame_manifest.csv` gives each frame's parsed trajectory ID and leading filename index.
- `trajectory_order_summary.csv` reports missing indices, duplicate indices, unparsed names, and order ambiguity. A numeric gap indicates missing observed filenames, not a missing physical state.

## What is deliberately not claimed

No model-based frequency is reported for “starts at 1x1,” “HTR comes last,” or “1x1 passes through bad.” A frozen SimCLR encoder is a representation, not a trained reconstruction-type classifier, and the short-run checkpoint policy leaves no saved downstream classifier to score frames. In particular, the current classifier setup has no validated `bad` state detector, so the third rule cannot be evaluated honestly from its outputs.

When the physics team confirms rule applicability and a saved, validated downstream classifier is available, a separate prediction audit can score trajectories. That future audit must report ambiguous ordering and violations, while keeping metadata fusion separate from whole-trajectory constraints.

# SimCLR pretraining provenance audit

Canonical repository: `https://github.com/Purpleline-z/rl_quantum_2026summer`  
Audited upstream repository: `https://github.com/ymeng3/Quantum`  
Pinned upstream commit: `bc19e167586da25ac314b5aa7c116e8d48141723`  
Audit date: 2026-08-09

The complete Git tree at the pinned commit was inspected through GitHub's
read-only tree API. It contains the pretrained encoder checkpoint at
`Classifier1/artifacts/encoders/simclr_resnet18_encoder.pth` and the downstream
files `Classifier2/train_unified.py`, `Classifier2/train_fraction.py`, and
`Classifier2/train_v5.7.py`. No executable SimCLR/self-supervised pretraining
entry point or its local dependencies are present in that commit.

Therefore no source file is vendored here. The downstream scripts load the
encoder checkpoint but are Bradley--Terry pairwise training scripts, not SimCLR
pretraining code. A SimCLR source file can be added only after its exact upstream
path and commit are supplied. This audit must not be treated as proof that the
upstream checkpoint produced the local shipped checkpoint.

Local compatibility check: the shipped checkpoint has 120 tensor keys beginning
with `0.weight`, `1.weight`, and related sequential ResNet-18 keys. This layout
matches the `nn.Sequential(*list(resnet18.children())[:-1])` encoder consumed by
the downstream scripts and current active-learning model. It does not establish
how the checkpoint was trained.

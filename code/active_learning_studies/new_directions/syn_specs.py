"""Learner specifications of the synthetic-anchor study (development + frozen candidates); imported by syn_run.py."""
import nd_synthetic as Y

T4 = ("1x1", "c6x2", "RT13", "HTR"); T3 = ("1x1", "c6x2", "RT13"); T2 = ("1x1", "c6x2")


def syn(bank, types, share, nodes=0, **kw):
    return dict(bank=bank, types=types, share=share, nodes=nodes, per_step=16, **kw)


# development round 2: calibrated bank v1 (simulator), augmented-real control bank augreal1
for name, kw in {
    "sy_v1_a25_T4": dict(syn=syn("v1", T4, 0.25)), "sy_v1_a50_T4": dict(syn=syn("v1", T4, 0.5)), "sy_v1_a25_T2": dict(syn=syn("v1", T2, 0.25)), "sy_v1_a25_T3": dict(syn=syn("v1", T3, 0.25)),
    "sy_v1_n100_T4": dict(syn=syn("v1", T4, 0.0, nodes=100)), "sy_v1_a25n100_T4": dict(syn=syn("v1", T4, 0.25, nodes=100)),
    "sy_v1_a25_T4_perm": dict(syn=syn("v1", T4, 0.25, perm=True)),
    "sy_aug_a25": dict(syn=syn("augreal1", T4, 0.25)),
}.items(): Y.register(name, **kw)

# development round 3: the nearby variants (masked real anchors, synthetic images only as unlabelled graph nodes)
for name, kw in {
    "sy_drop16": dict(drop16=True), "sy_mask_drop16": dict(ideal="mask", drop16=True),
    "sy_mask_n100": dict(ideal="mask", syn=syn("v1", T4, 0.0, nodes=100)), "sy_v1_n300_T4": dict(syn=syn("v1", T4, 0.0, nodes=300)), "sy_v2_n100_T4": dict(syn=syn("v2", T4, 0.0, nodes=100)),
    "sy_aug_n100": dict(syn=syn("augreal1", T4, 0.0, nodes=100)), "sy_v1_n100_T3": dict(syn=syn("v1", T3, 0.0, nodes=100)),
}.items(): Y.register(name, **kw)

# development round 4: two-head combinations (ensemble / anchor-proximity gate); control = two baseline heads with different initialisation
base_spec = Y.make_spec("base"); mask_spec = Y.make_spec("mask", ideal="mask"); maskd_spec = Y.make_spec("maskd", ideal="mask", drop16=True)
Y.register_two("sy_ens_base2", base_spec, base_spec, "mean", seed_b_offset=7777)
Y.register_two("sy_ens_mask", base_spec, mask_spec, "mean"); Y.register_two("sy_gate_mask", base_spec, mask_spec, "gate"); Y.register_two("sy_gate_maskd", base_spec, maskd_spec, "gate")
Y.register_two("sy_gate_base2", base_spec, base_spec, "gate", seed_b_offset=7777)   # control of the gate: same gating with a second baseline head

# development round 5: type-mean-aligned banks (real type centres + simulator variation) and their controls (Gaussian with the same per-type mean / covariance; permuted types; augmented real images)
for name, kw in {
    "sy_tm_a10": dict(syn=syn("v1_typemean", T4, 0.10)), "sy_tm_a25": dict(syn=syn("v1_typemean", T4, 0.25)), "sy_tm_a50": dict(syn=syn("v1_typemean", T4, 0.50)),
    "sy_tm_a25_perm": dict(syn=syn("v1_typemean", T4, 0.25, perm=True)), "sy_tmg_a25": dict(syn=syn("v1_tmgauss", T4, 0.25)), "sy_tmg_a50": dict(syn=syn("v1_tmgauss", T4, 0.50)),
    "sy_aug_a50": dict(syn=syn("augreal1", T4, 0.50)),
}.items(): Y.register(name, **kw)

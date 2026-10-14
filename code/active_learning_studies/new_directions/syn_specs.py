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

"""Minimal KINEMATIC RHEED renderer (synthetic-anchor study).

Physics (and what is NOT modelled)
  * Surface unit cell a = 3.905 A (SrTiO3(001)); superlattices are given by primitive vectors (integer matrix, units of a): 1x1 = (1,0),(0,1); c(6x2) = (3,1),(3,-1) [centred cell, area 6];
    (sqrt13 x sqrt13)R33.7 = (3,2),(-2,3) [area 13, both enantiomers]; "HTR" is an arbitrary SURROGATE (the literature search found no lattice for the lab's "HTR"): 1x1 rods + weak half-order rods + strong Kikuchi
    + diffuse background.  c(6x2) and RT13 are rotational / mirror domain mixtures with random domain weights.
  * Reciprocal-lattice RODS (surface truncation) G = h B1 + k B2, B = 2 pi/a (S^-1)^T, rotated by the beam azimuth.  For every detector pixel (Y, Z) at distance L from the sample the scattering vector
    dK = k_f - k_i is computed exactly (k_i = k (cos t, 0, -sin t), k_f = k (L, Y, Z)/|.|); the intensity is sum_r w_r exp(-(dK_y - G_y)^2 / 2 s_y^2) exp(-(dK_x - G_x)^2 / 2 s_x^2):
    s_y = finite lateral domain size (streak width), s_x = Ewald-sphere thickness + along-beam rod thickness (streak length).  This yields streaks, Laue circles, higher-Laue-zone arcs and the
    specular / direct-beam geometry without extra assumptions.  Elastic, SINGLE scattering only: no dynamical / multiple scattering, no structure factors (rod weights are random surrogates),
    no inelastic (Kikuchi) physics beyond two phenomenological lines, no thermal diffuse scattering beyond a smooth background.
  * Detector: exposure gain, Poisson-like + Gaussian noise, blur, 8-bit quantisation, shadow edge, optional direct beam + halo (uncropped style), horizontal / vertical shifts.
Calibration constants come from the real images of this repository (65 px rod spacing, specular near (293, 225), shadow edge near y = 135 of the 656 x 492 frame; see syn_calibrate.py).
"""
from __future__ import annotations

import numpy as np

A_LATTICE = 3.905
E_KEV = 15.0
LAM = 12.2643 / np.sqrt(E_KEV * 1e3 * (1 + 0.97845e-6 * E_KEV * 1e3))   # Angstrom
K0 = 2 * np.pi / LAM
SPACING_PX = 65.0
L_PX = SPACING_PX * K0 / (2 * np.pi / A_LATTICE)           # detector distance in pixels so that 1x1 rods are 65 px apart
FULL = (492, 656)

LATTICES = {"1x1": [((1, 0), (0, 1))], "c6x2": [((3, 1), (3, -1))], "RT13": [((3, 2), (-2, 3)), ((2, 3), (3, -2))], "HTR": [((1, 0), (0, 1))]}
TYPES = ("1x1", "c6x2", "RT13", "HTR")


def rods(typ: str, rng, azimuth_deg: float, gmax: float = 7.0, super_weight=None, domain_alpha: float = 1.0, sup_scale: float = 1.0):
    """List of (Gx, Gy, weight) of the visible rods in the beam frame (x along the beam, y lateral)."""
    out = []; phi = np.deg2rad(azimuth_deg)
    sw = sup_scale * (super_weight if super_weight is not None else {"1x1": 0.0, "c6x2": rng.uniform(0.04, 0.4), "RT13": rng.uniform(0.03, 0.35), "HTR": rng.uniform(0.0, 0.2)}[typ])
    sets = []
    for S in LATTICES[typ]:
        Sm = np.array(S, float); B = 2 * np.pi / A_LATTICE * np.linalg.inv(Sm).T
        for rot in ((0, 90, 180, 270) if typ in ("c6x2", "RT13") else (0,)):
            sets.append((B, np.deg2rad(rot)))
    dom_w = rng.dirichlet(np.full(len(sets), domain_alpha)) * len(sets) if len(sets) > 1 else np.ones(1)
    fund = 2 * np.pi / A_LATTICE; seen = {}
    for (B, rot), dw in zip(sets, dom_w):
        c, s = np.cos(rot + phi), np.sin(rot + phi); R = np.array([[c, -s], [s, c]])
        for h in range(-40, 41):
            for k in range(-40, 41):
                g = h * B[0] + k * B[1]
                if np.hypot(*g) > gmax * fund / 2: continue
                gx, gy = R @ g
                if gx > 0.12 or gx < -0.9 or abs(gy) > 4.5 * fund / 1.0: continue          # only rods that can meet the Ewald sphere on the screen
                # fundamental rods (integer order of the 1x1 lattice) have weight 1, superstructure rods the surface weight
                gfund = np.linalg.solve(np.array([[fund, 0], [0, fund]]), g); fundamental = np.allclose(gfund, np.round(gfund), atol=1e-6)
                w = (1.0 if fundamental else sw * dw) * np.exp(-(np.hypot(*g) / (2.2 * fund)) ** 2) * (np.exp(rng.normal(0, 0.45)) if not fundamental else 1.0)
                if typ == "HTR" and not fundamental: w *= 0.5
                key = (round(float(gx), 4), round(float(gy), 4)); seen[key] = seen.get(key, 0.0) + w
    return [(gx, gy, w) for (gx, gy), w in seen.items() if w > 1e-3]


TUNE = {"sy": 1.0, "sx": 1.0, "sup": 1.0, "spec": 1.0, "gain": 1.0, "bg": 1.0, "blur": 1.0, "kik": 1.0, "noise": 1.0, "rodw": 1.0}   # calibration multipliers (syn_calibrate.py)


def sample_params(rng, typ: str, style: str = "traj") -> dict:
    """Domain-randomised parameters; ``style``: 'traj' (trajectory-frame crop: shadow region dark, no direct beam) or 'ideal' (uncropped, direct beam + halo visible)."""
    p = dict(typ=typ, style=style)
    p["theta"] = np.deg2rad(rng.uniform(1.7, 2.3)); p["x0"] = rng.normal(293.0, 5.0); p["yh"] = rng.uniform(128.0, 146.0)
    p["azimuth"] = rng.normal(0.0, 0.5); p["sig_y_px"] = rng.uniform(3.2, 6.5); p["sig_x"] = rng.uniform(0.022, 0.05)
    p["super_weight"] = None; p["domain_alpha"] = rng.choice([0.4, 1.0, 3.0])
    p["gain"] = float(np.exp(rng.normal(np.log(1.0), 0.5))); p["bg"] = rng.uniform(0.02, 0.15); p["bg_decay"] = rng.uniform(80, 300); p["blur"] = rng.uniform(0.8, 2.2)
    p["spec"] = rng.uniform(0.5, 4.0); p["spec_px"] = rng.uniform(4.0, 8.0); p["kikuchi"] = rng.uniform(0.0, 0.07) * (3.0 if typ == "HTR" else 1.0); p["kik_slope"] = rng.uniform(0.5, 1.1)
    p["noise_read"] = rng.uniform(0.5, 2.5); p["shot"] = rng.uniform(0.02, 0.12); p["dx"] = rng.normal(0, 3.0); p["dy"] = rng.normal(0, 3.0); p["sheen"] = rng.uniform(0.0, 0.4)
    p["rod_mod"] = rng.uniform(0.0, 0.3); p["rod_mod_phase"] = rng.uniform(0, 2 * np.pi)
    p["sig_y_px"] *= TUNE["sy"]; p["sig_x"] *= TUNE["sx"]; p["gain"] *= TUNE["gain"]; p["bg"] *= TUNE["bg"]; p["blur"] *= TUNE["blur"]; p["spec"] *= TUNE["spec"]; p["kikuchi"] *= TUNE["kik"]; p["noise_read"] *= TUNE["noise"]; p["sup_scale"] = TUNE["sup"]; p["rod_scale"] = TUNE["rodw"]
    return p


def render(p: dict, rng, half: bool = True) -> np.ndarray:
    """uint8 image of FULL size (656 x 492); rendered on a half-resolution grid (physical pixel units stay those of the full frame) and bicubically up-sampled when ``half``."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    H, W = (FULL[0] // 2, FULL[1] // 2) if half else FULL; step = 2.0 if half else 1.0
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32) * step + step / 2
    Y = cc - (p["x0"] + p["dx"]); Z = rr - (p["yh"] + p["dy"])                  # Z > 0: below the shadow edge (screen), pattern side
    th = p["theta"]; R = np.sqrt(L_PX ** 2 + Y ** 2 + Z ** 2)
    kfx, kfy, kfz = K0 * L_PX / R, K0 * Y / R, K0 * Z / R
    dKx = kfx - K0 * np.cos(th); dKy = kfy
    sy = p["sig_y_px"] * K0 / L_PX; sx = p["sig_x"]
    I = np.zeros((H, W), np.float32)
    dKz = kfz + K0 * np.sin(th)
    for gx, gy, w in rods(p["typ"], rng, p["azimuth"], super_weight=p["super_weight"], domain_alpha=p["domain_alpha"], sup_scale=p.get("sup_scale", 1.0)):
        if gx - 4 * sx > dKx.max() or gx + 4 * sx < dKx.min(): continue
        I += (w * p.get("rod_scale", 1.0) * np.exp(-0.5 * ((dKx - gx) / sx) ** 2 - 0.5 * ((dKy - gy) / sy) ** 2)).astype(np.float32)
    I *= (1 + p["rod_mod"] * np.cos(dKz * A_LATTICE * 1.0 + p["rod_mod_phase"])).astype(np.float32) / (1 + p["rod_mod"])      # 3-D modulation along the rods (spots along the streaks)
    I *= np.clip(Z / 12.0, 0, 1) ** 1.5                                             # shadow edge: the sample blocks everything below the horizon (screen: above the edge)
    zs = (L_PX * np.tan(th)); spec = p["spec"] * np.exp(-0.5 * ((Y / p["spec_px"]) ** 2 + ((Z - zs) / (p["spec_px"] * 1.4)) ** 2))
    I += spec.astype(np.float32) * np.clip(Z / 12.0, 0, 1)
    bgp = p["bg"] * np.exp(-np.clip(Z, 0, None) / p["bg_decay"]) * np.clip(Z / 10.0, 0, 1) * (1 + p["sheen"] * np.exp(-0.5 * (Y / 150.0) ** 2))
    I += bgp.astype(np.float32)
    if p["kikuchi"] > 0:    # two phenomenological Kikuchi lines through the specular region (V shape)
        for sgn in (-1, 1):
            d = (Y - sgn * (Z - zs) * p["kik_slope"]) / np.sqrt(1 + p["kik_slope"] ** 2)
            I += (p["kikuchi"] * np.exp(-0.5 * (d / 3.5) ** 2) * np.exp(-np.abs(Z - zs) / 160.0) * np.clip(Z / 12.0, 0, 1)).astype(np.float32)
    if p["style"] == "ideal":   # direct beam + halo ring above the shadow edge (uncropped frames)
        zd = -zs; db = 3.0 * np.exp(-0.5 * ((Y / 5.0) ** 2 + ((Z - zd) / 7.0) ** 2)) + 0.05 * np.exp(-0.5 * ((np.sqrt(Y ** 2 + (Z - zd) ** 2) - 45.0) / 5.0) ** 2); I += db.astype(np.float32)
    I = gaussian_filter(I, p["blur"] / step)
    I = I * p["gain"] * 160.0
    I = rng.poisson(np.clip(I * p["shot"] * 10, 0, 1e5)).astype(np.float32) / (p["shot"] * 10) if p["shot"] > 0 else I
    I = I + rng.normal(0, p["noise_read"], I.shape)
    img = np.clip(I, 0, 255).astype(np.uint8)
    if half: img = np.array(Image.fromarray(img, "L").resize((FULL[1], FULL[0]), Image.BICUBIC))
    return img


def generate(typ: str, n: int, seed: int, style: str = "traj"):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        p = sample_params(rng, typ, style); out.append((render(p, rng), p))
    return out

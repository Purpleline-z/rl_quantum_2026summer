# Physics-informed RHEED descriptors — documentation for the lab's physicists
Code: `new_directions/pf_extract.py` (extractor), `pf_build.py` (runs it on every image, 90 s on 4 cores), `nd_physics.py` (turns descriptors into head inputs / graph metrics), `pf_figures.py` (the figures in `figures_pf/`). No learning, no training labels, only numpy / scipy; deterministic.

## What the extractor does (per image)
1. **Decoding.** 8-bit palette images: PIL `convert("L")` (as the whole repository). The 18 ideal images stored as 16-bit PNG (780x580, other camera set-up with the viewport edge visible; 10 HTR, 8 RT13) are divided by 257 (brightness scale of the 8-bit images) and resampled to the common 656x492 frame. *The repository's own loaders clip these at 255 (saturated discs); see RESULTS section "data audit".*
2. **Horizon** (sample shadow edge): the row with the strongest dark→bright step of the row-median of the central columns (searched between 5 % and 70 % of the height). Everything above it (direct beam spot, Kikuchi-free dark region) is ignored.
3. **Specular spot S = (row, x0)**: centroid of the brightest compact blob of the Gaussian-smoothed (σ = 4 px) image below the horizon, central 60 % of the columns. x0 is the centre of the pattern (the (0,0) rod). Verified on contact sheets (`figures_pf/detection_overlays.png`); the detector is least reliable on the 16-bit images (viewport geometry) and on frames with the specular spot at the horizon.
4. **Five horizontal bands** relative to the specular row (rows r + a0 … r + a1, clipped to stay below the horizon): a = −45…−10 (rods just above the specular spot), b = −10…+10 (specular row), c = +15…+60, d = +60…+130, e = +130…+230 (lower region with the spot arrays of the reconstructions).
5. **Profile** p(x) = mean intensity of the band rows, x = column offset from x0 (±240 px). **Baseline**: p minus its 20th-percentile filter (61 px window) = diffuse background / Kikuchi / brightness gradient removed; **amplitude normalisation**: divide by the 99.5th percentile of the corrected profile (brightness invariance; the absolute level is kept separately in the background block). **Folding**: (p(x) + p(−x)) / 2, binned (mirror symmetry of the pattern; robust to a few pixels of x0 error).
6. **Integer-order spacing a**: position (25…110 px) of the strongest peak of the folded band-a profile = first-order rod. (Per image; it makes the "u" blocks independent of camera zoom / beam energy; it is by construction a peak at u = 1.)

## Descriptor blocks
| block | dim | content | physical meaning |
|---|---|---|---|
| `pxprof` | 120 | folded, normalised profiles of bands a–d, 30 bins of 8 px | line cuts in pixel units: positions and widths of all rods |
| `uprof` | 120 | same, sampled at u = 0.1 … 3.0 in units of a (max over ±0.03 u) | rod pattern in units of the integer spacing: fractional orders |
| `peaks` | 88 | heights at u ∈ {1/13, 2/13, 3/13, 1/6, 1/4, 1/3, 4/13, 5/13, 1/2, 6/13, 7/13, 2/3, 3/4, 5/6, 1, 7/6, 4/3, 3/2, 5/3, 2, 5/2, 3}, bands a–d, relative to the first-order rod (band b: relative to the specular rod) | fractional-order peak heights: (1x1) → none; c(6x2) → 1/6, 1/3, 1/2 …; root13 → multiples of 1/13-related positions (not an established assignment, the search found no source for it) |
| `fourier` | 21 | amplitude of the Hann-windowed profile at spatial frequencies m / a, m ∈ {1, 2, 3, 6, 13/2, 13/3, 13}, bands a, b, d | periodicities 1/2, 1/3, 1/6 … of the pattern |
| `bg` | 13 | log amplitude and log mean of the baseline-corrected profile per band, log mean level above / below the horizon, specular row minus horizon | brightness and background; absolute intensity cues |
| `width` | 3 | FWHM-like width, integral and peak of the specular rod in band b | rod width (surface roughness / domain size) |
| `spacing` | 1 | a / 62 px | integer spacing (zoom) |
Feature sets used by the learners: `pfa` = pxprof + bg + width (136-d); `pfb` = peaks + fourier + bg + width + spacing (126-d); `pfc` = all (366-d). Learners standardise with the z-scores of the context's allowed graph nodes (clipped at ±5), then scale to the norm of the SimCLR feature.

## What to inspect
`figures_pf/detection_overlays.png` (red = horizon, green = specular spot, blue = bands), `figures_pf/profiles_by_type.png` (mean ± sd profiles in units of a per type and band), `figures_pf/pca_physics.png`, validation table in `RESULTS_physics_features.md`. The descriptors are *not* a physical analysis (no peak fitting, no Ewald-sphere geometry, no calibration of pixels to reciprocal-lattice units); the integer spacing is estimated from the image, which removes zoom but also assumes that the strongest rod in band a is first order.

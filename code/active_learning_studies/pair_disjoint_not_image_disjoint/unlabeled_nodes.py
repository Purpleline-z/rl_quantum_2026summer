"""Unlabelled trajectory images as extra nodes of the image graph (label-free, identity-safe by construction).

Source: the cached SimCLR features of all 1278 images (`image_representation_analysis/results/representation_exploration/cache/rheed_simclr_resnet18.npy`; for the 444 images that also
live in `simclr_feature_cache.pt` the vectors are identical, cosine >= 0.99999).  An image is an unlabelled node only if
  1. its path is in no pairwise-comparison row of the label sheet (any type, any version of the row), and it is not an ideal image; and
  2. its feature vector is not (near-)identical (cosine of mean-centred vectors > 0.9999) to any image of the label sheet or any ideal image,
     i.e. no byte-identical copy of a pair image, test image, utility image or reference image can enter the graph under another file name.
Result: 509 of the 824 candidate trajectory images remain (331 were near-duplicates; ideal images are partly copies of trajectory frames).
Nodes carry no label; they only change the graph (kNN edges, label propagation of the typed reference images, SGC smoothing)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REP = ROOT / "code/active_learning_studies/image_representation_analysis/results/representation_exploration"
SHEET = ROOT / "data/original data/Quantum Label Data - Pairwise_Comparisonv1.8.csv"
DUPLICATE_COSINE = 0.9999


@lru_cache(maxsize=1)
def load_unlabeled() -> dict[str, np.ndarray]:
    """name -> 512-d SimCLR feature of every admissible unlabelled trajectory image; names are 'unlabelled::<manifest path>'."""
    if not (REP / "cache/rheed_simclr_resnet18.npy").exists() or not SHEET.exists(): return {}
    man = pd.read_csv(REP / "manifest.csv"); man["key"] = man.path.str.replace(r"^.*/(original data/.*)$", r"\1", regex=True)
    x = np.load(REP / "cache/rheed_simclr_resnet18.npy").astype(np.float64); idx = {k: i for i, k in enumerate(man.key)}
    sheet = pd.read_csv(SHEET); used = {idx[k] for k in ("original data/" + sheet.Image1_Path).tolist() + ("original data/" + sheet.Image2_Path).tolist() if k in idx}
    ideal = set(man.index[man.source == "ideal"]); z = x - x.mean(0); z /= np.linalg.norm(z, axis=1, keepdims=True)
    protected = sorted(used | ideal); keep = []
    for i in man.index[man.source == "trajectory"]:
        if i in used: continue
        if float((z[protected] @ z[i]).max()) > DUPLICATE_COSINE: continue
        keep.append(int(i))
    return {f"unlabelled::{man.key[i]}": x[i].copy() for i in keep}

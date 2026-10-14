"""HTR failure analysis, step 0: static tables of every judgment and image (independent of any learner).

  python3 hf_table.py   ->  results/hf/judgments.csv (one row per usable judgment, 521), results/hf/images.csv (one row per image used in a judgment)

Image columns: run, session, frame index (position in the growth run), temperature, nearest ideal-image class (cosine on centred SimCLR features, k = 1 and the 5-NN vote), distance to the nearest ideal image of every class,
and the 1 - cos distance to the nearest other judged image (near-duplicate measure).  Judgment columns: pair, type, winner, labeller, confidence, the two images' frame/temperature, time gap inside a run, feature distance between the images.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nd_core as core
import rt_core as rt

OUT = core.HERE / "results" / "hf"
NAME = re.compile(r"(\d+)_(RR\d+[A-Z]?)_(\d+)C?_(\d+)")
CLASSES = ["1x1", "c6x2", "RT13", "HTR", "Twinned"]; ACTIVE_CLASSES = CLASSES[:4]   # Twinned has 4 ideal images and is excluded from the learner


def ideal_class(p: str):
    m = re.search(r"STO_ideal_([A-Za-z0-9]+)", p)
    if not m: return None
    return {"1x1": "1x1", "c6x2": "c6x2", "RT13": "RT13", "HTR": "HTR", "Twinned2x1": "Twinned"}[m.group(1)]


def frame_info(p: str) -> dict:
    m = NAME.search(Path(p).name)
    if "Trajectories" not in p or not m: return {"kind": "ideal", "run": None, "idx": np.nan, "temp": np.nan, "frame": np.nan}
    return {"kind": "traj", "run": m.group(2), "idx": int(m.group(1)), "temp": float(m.group(3)), "frame": int(m.group(4))}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(core.HF_DEV_SEEDS[0], "A", scratch, cache)
    rows = pd.concat(ctx.exp.groups.values()).reset_index(drop=True); root = str(core.single_study.DATA.resolve()) + "/original data/"
    rel = lambda p: str(p).replace(root, "")
    images = sorted({rel(p) for p in list(rows.resolved_img1) + list(rows.resolved_img2)}); full = {rel(p): str(p) for p in list(rows.resolved_img1) + list(rows.resolved_img2)}
    ideals = {}
    for p in cache:
        c = ideal_class(str(p))
        if c: ideals[str(p)] = c
    ideal_paths = sorted(ideals); ideal_cls = np.array([ideals[p] for p in ideal_paths])
    img = pd.DataFrame({"img": images}); info = pd.DataFrame([frame_info(i) for i in images]); img = pd.concat([img, info], axis=1)
    xs = sim.xn[[sim.index[full[i]] for i in images]]; xi = sim.xn[[sim.index[p] for p in ideal_paths]]
    s = (xs @ xi.T).numpy()
    for i, im in enumerate(images):   # an ideal image must not be its own nearest neighbour
        if full[im] in ideals: s[i, ideal_paths.index(full[im])] = -2
    for c in CLASSES:
        sel = ideal_cls == c
        if sel.any(): img[f"d_{c}"] = 1 - s[:, sel].max(1)
    img["nn_class"] = [ACTIVE_CLASSES[int(np.argmin([r[f'd_{c}'] for c in ACTIVE_CLASSES]))] for _, r in img.iterrows()]
    s4 = np.where(np.isin(ideal_cls, ACTIVE_CLASSES)[None, :], s, -2); k5 = np.argsort(-s4, axis=1)[:, :5]; img["vote5"] = [pd.Series(ideal_cls[r]).value_counts().index[0] for r in k5]
    img["d_ideal_min"] = np.minimum.reduce([img[f"d_{c}"] for c in ACTIVE_CLASSES])
    ss = (xs @ xs.T).numpy(); np.fill_diagonal(ss, -2); img["d_nearest_judged"] = 1 - ss.max(1)
    img["img_id"] = np.arange(len(img)); dm = 1 - ss; np.fill_diagonal(dm, 0.0); OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "img_dist.npy", dm)
    ix = {im: i for i, im in enumerate(images)}
    j = pd.DataFrame({"pair": [rel(p) for p in rows.pair_id], "type_idx": rows.type_idx.astype(int), "type": rows.canonical_type, "winner": rows.Winner, "labeller": rows.Labeler_Name,
                      "confidence": rows.Confidence.fillna("none"), "w": rows.confidence_weight, "created": rows.Created_At,
                      "img1": [rel(p) for p in rows.resolved_img1], "img2": [rel(p) for p in rows.resolved_img2]})
    j["i1"], j["i2"] = j.img1.map(ix), j.img2.map(ix)
    for side in (1, 2):
        for c in ("run", "idx", "temp", "frame", "nn_class", "vote5", "d_ideal_min"): j[f"{c}{side}"] = img.loc[j[f"i{side}"], c].to_numpy()
        for c in CLASSES:
            if f"d_{c}" in img: j[f"d_{c}{side}"] = img.loc[j[f"i{side}"], f"d_{c}"].to_numpy()
    j["same_run"] = j.run1 == j.run2; j["frame_gap"] = np.where(j.same_run, (j.idx1 - j.idx2).abs(), np.nan); j["temp_gap"] = (j.temp1 - j.temp2).abs()
    j["d_pair"] = 1 - ss_pair(sim, [full[i] for i in j.img1], [full[i] for i in j.img2])
    j["decisive"] = j.winner.isin(["1", "2"]); j["sign"] = np.where(j.winner == "1", 1.0, np.where(j.winner == "2", -1.0, 0.0))
    j["cls_pair"] = [("-".join(sorted([a, b]))) for a, b in zip(j.nn_class1, j.nn_class2)]
    j["jid"] = np.arange(len(j)); return j, img


def ss_pair(sim, a, b) -> np.ndarray:
    ia = [sim.index[x] for x in a]; ib = [sim.index[x] for x in b]; return (sim.xn[ia] * sim.xn[ib]).sum(1).numpy()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); j, img = build(); j.to_csv(OUT / "judgments.csv", index=False); img.to_csv(OUT / "images.csv", index=False); print(len(j), len(img))

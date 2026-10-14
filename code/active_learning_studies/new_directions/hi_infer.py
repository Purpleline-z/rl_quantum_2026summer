"""Group-level (cluster) inference for paired learner comparisons from per-judgment held-out predictions (honest-inference study).  numpy / pandas / scipy only (no torch).

Data (written by hi_run.py): per (seed, split) a meta file (one row per held-out decisive judgment: pair group, type, sign, confidence weight, distance to the nearest typed anchor) and a long prediction file
(budget, draw, learner, j, d).  A CELL is (seed, split, budget, draw): one random labelled set, identical for every learner, evaluated on the held-out judgments of that seed/split.

Estimator ("standard endpoint", unchanged): per cell the metric of the candidate minus that of the reference (AUC, accuracy, log-loss with sign so that positive = candidate better; accuracy and log-loss
are confidence-weighted, AUC unweighted, exactly as nd_core.metrics_from_d), averaged over the cells of a split.  RESAMPLED estimator: every held-out judgment gets the weight of the resampling unit of its
pair group (the number of times the unit is drawn), inside every cell; metrics are weighted (AUC: weighted Mann-Whitney with ties 1/2); cells in which a resample leaves no positive or no negative judgment
are skipped (nan-mean).  Units: pair groups (primary), or clusters of near-duplicate images / coarse blocks (hi_clusters.py).  Percentile intervals; p-values by sign-flip permutation of per-unit contributions.

Per-unit contributions A_u (exact additive decomposition of the point estimate: sum_u A_u = estimate): accuracy / log-loss: the weighted judgment gains of the unit's judgments divided by the cell's total
weight, summed over cells and divided by the number of cells; AUC: each (positive, negative) pair's gain C^cand_ij - C^ref_ij divided by P_c N_c, half to the unit of i and half to the unit of j.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
METRICS = ("auc", "acc", "ll")


# ------------------------------------------------------------------------------------------------ data
class Block:
    """All cells of one or more result directories (hi_run.py output) as flat arrays; cells are contiguous slices."""
    def __init__(self, dirs, clusters=None, learners=None):
        dirs = [dirs] if isinstance(dirs, (str, Path)) else list(dirs); cl = pd.read_csv(clusters or HERE / "results" / "hi" / "clusters.csv"); self.clusters = cl; gindex = {g: i for i, g in enumerate(cl.group)}
        cells, arrays = [], {k: [] for k in ("grp", "k", "sign", "wt", "anc", "anc_ref")}; D = {}; start = 0
        for d in dirs:
            for pf in sorted(glob.glob(str(Path(d) / "pred" / "seed*.csv.gz"))):
                stem = Path(pf).name.replace(".csv.gz", ""); seed, split = int(stem[4:].split("_")[0]), stem.split("_")[1]
                meta = pd.read_csv(Path(d) / "meta" / f"{stem}.csv"); pred = pd.read_csv(pf); gi = np.array([gindex[g] for g in meta.group]); nj = len(meta)
                for (budget, draw), sub in pred.groupby(["budget", "draw"], sort=True):
                    wide = sub.pivot(index="j", columns="learner", values="d").sort_index()
                    if learners is not None and not set(learners) <= set(wide.columns): continue
                    assert len(wide) == nj
                    cells.append((seed, split, int(budget), int(draw), start, start + nj)); start += nj
                    arrays["grp"].append(gi); arrays["k"].append(meta.k.to_numpy()); arrays["sign"].append(meta.sign.to_numpy()); arrays["wt"].append(meta.weight.to_numpy()); arrays["anc"].append(meta.anc_all_min.to_numpy()); arrays["anc_ref"].append(meta.anc_ref_min.to_numpy())
                    for l in wide.columns: D.setdefault(l, []).append(wide[l].to_numpy())
        self.cells = pd.DataFrame(cells, columns=["seed", "split", "budget", "draw", "start", "stop"]); self.n_groups = len(cl)
        for k, v in arrays.items(): setattr(self, k, np.concatenate(v))
        self.D = {l: np.concatenate(v) for l, v in D.items() if len(v) == len(cells)}; self.cell_of = np.repeat(np.arange(len(cells)), self.cells.stop - self.cells.start)

    def learners(self): return sorted(self.D)

    def metric_arrays(self, name):
        d, s = self.D[name], self.sign
        return {"acc": ((d * s) > 0).astype(float), "ll": np.logaddexp(0.0, -s * d)}


# ------------------------------------------------------------------------------------------------ padded per-cell design of a comparison
class Design:
    """Per cell c and per group slot g (the groups present in the cell's selected judgments): for the ratio estimators
       acc/ll:  num[c, g] = sum_{j in g} wt_j * gain_j,  den[c, g] = sum_{j in g} wt_j          (metric_c = sum_g n_g num / sum_g n_g den)
       auc:     Q[c, g, h] = sum_{i in g pos, j in h neg} (C^cand_ij - C^ref_ij),  R[c, g, h] = #pos_g * #neg_h   (metric_c = n^T Q n / n^T R n),  with C = 1[d_i > d_j] + 1/2 1[d_i = d_j].
       n = unit multiplicity of the group's unit.  Cells with an empty selection are dropped."""
    def __init__(self, blk: Block, cand: str, ref: str, metric: str, mask: np.ndarray | None = None, cell_mask: np.ndarray | None = None):
        self.metric = metric; self.blk = blk; keep_cells = np.ones(len(blk.cells), bool) if cell_mask is None else cell_mask; gl, mats = [], []
        if metric != "auc": ma, mr = blk.metric_arrays(cand)[metric], blk.metric_arrays(ref)[metric]; gain = (ma - mr) if metric == "acc" else (mr - ma)
        for c, (st, sp) in enumerate(zip(blk.cells.start, blk.cells.stop)):
            if not keep_cells[c]: continue
            sel = np.arange(st, sp) if mask is None else np.arange(st, sp)[mask[st:sp]]
            if len(sel) == 0: continue
            g = blk.grp[sel]; ug, inv = np.unique(g, return_inverse=True)
            if metric != "auc":
                w = blk.wt[sel]; num = np.bincount(inv, w * gain[sel], len(ug)); den = np.bincount(inv, w, len(ug)); mats.append((c, ug, num, den))
            else:
                pos, neg = blk.sign[sel] > 0, blk.sign[sel] <= 0
                if pos.sum() == 0 or neg.sum() == 0: continue
                dc, dr = blk.D[cand][sel], blk.D[ref][sel]
                Ca = (dc[pos][:, None] > dc[neg][None, :]) + 0.5 * (dc[pos][:, None] == dc[neg][None, :]); Cr = (dr[pos][:, None] > dr[neg][None, :]) + 0.5 * (dr[pos][:, None] == dr[neg][None, :])
                Gp = np.zeros((pos.sum(), len(ug))); Gp[np.arange(pos.sum()), inv[pos]] = 1; Gn = np.zeros((neg.sum(), len(ug))); Gn[np.arange(neg.sum()), inv[neg]] = 1
                Cab = (dc[pos][:, None] > dr[neg][None, :]) + 0.5 * (dc[pos][:, None] == dr[neg][None, :]); Cba = (dr[pos][:, None] > dc[neg][None, :]) + 0.5 * (dr[pos][:, None] == dc[neg][None, :])   # cross-learner pair values
                Q = Gp.T @ (Ca - Cr) @ Gn; Qx = Gp.T @ (Cab - Cba) @ Gn; R = np.outer(Gp.sum(0), Gn.sum(0)); mats.append((c, ug, Q, R, Qx))
        self.cell_ids = np.array([m[0] for m in mats]); self.nc = len(mats); G = max((len(m[1]) for m in mats), default=1); self.G = G
        self.gidx = -np.ones((self.nc, G), int)
        if metric == "auc": self.Q = np.zeros((self.nc, G, G)); self.R = np.zeros((self.nc, G, G)); self.Qx = np.zeros((self.nc, G, G))
        else: self.num = np.zeros((self.nc, G)); self.den = np.zeros((self.nc, G))
        for r, m in enumerate(mats):
            k = len(m[1]); self.gidx[r, :k] = m[1]
            if metric == "auc": self.Q[r, :k, :k] = m[2]; self.R[r, :k, :k] = m[3]; self.Qx[r, :k, :k] = m[4]
            else: self.num[r, :k] = m[2]; self.den[r, :k] = m[3]
        self.cell_split = blk.cells.split.to_numpy()[self.cell_ids]; self.cell_seed = blk.cells.seed.to_numpy()[self.cell_ids]; self.cell_budget = blk.cells.budget.to_numpy()[self.cell_ids]; self.cell_draw = blk.cells.draw.to_numpy()[self.cell_ids]

    # -- cell metrics for unit-multiplicity matrices X (B, n_groups) over GROUPS (already mapped from units)
    def cell_values(self, X: np.ndarray) -> np.ndarray:
        Xc = np.where(self.gidx[None] >= 0, X[:, np.clip(self.gidx, 0, None)], 0.0)             # (B, nc, G)
        if self.metric == "auc":
            num = np.einsum("bcg,cgh,bch->bc", Xc, self.Q, Xc, optimize=True); den = np.einsum("bcg,cgh,bch->bc", Xc, self.R, Xc, optimize=True)
        else: num = (Xc * self.num[None]).sum(2); den = (Xc * self.den[None]).sum(2)
        with np.errstate(invalid="ignore", divide="ignore"): return np.where(den > 0, num / den, np.nan)

    def point(self) -> np.ndarray: return self.cell_values(np.ones((1, self.blk.n_groups)))[0]
    def estimate(self) -> float: return float(np.nanmean(self.point()))

    def contributions(self) -> np.ndarray:
        """A_g per group (length n_groups): additive decomposition of estimate() (cells with a defined metric only)."""
        a = np.zeros(self.blk.n_groups); ok = np.isfinite(self.point()); ncell = ok.sum()
        for r in np.where(ok)[0]:
            ids = self.gidx[r]; v = ids >= 0; ids = ids[v]
            if self.metric == "auc":
                Q = self.Q[r][np.ix_(v, v)]; tot = self.R[r].sum(); np.add.at(a, ids, 0.5 * (Q.sum(1) + Q.sum(0)) / tot / ncell)
            else: np.add.at(a, ids, self.num[r][v] / self.den[r][v].sum() / ncell)
        return a


    def perm_contributions(self) -> np.ndarray:
        """B_g (length n_groups) for the exact group-level randomisation test of exchangeability of candidate and reference: swapping the two learners' predictions for a set of groups changes the estimate to
        sum_g e_g B_g with e_g = +1 (kept) / -1 (swapped).  accuracy / log-loss: B = A.  AUC: B_g = A_g + (cross term): a pair (i in g, j in h) contributes (e_g + e_h)/2 * (C^cand_ij - C^ref_ij)
        + (e_g - e_h)/2 * (C(d^cand_i, d^ref_j) - C(d^ref_i, d^cand_j)) once the learners are swapped for part of the groups (the cross-learner comparisons).  sum_g B_g = estimate."""
        if self.metric != "auc": return self.contributions()
        b = np.zeros(self.blk.n_groups); ok = np.isfinite(self.point()); ncell = ok.sum()
        for r in np.where(ok)[0]:
            ids = self.gidx[r]; v = ids >= 0; ids = ids[v]; Q = self.Q[r][np.ix_(v, v)]; X = self.Qx[r][np.ix_(v, v)]; tot = self.R[r].sum()
            np.add.at(b, ids, (0.5 * (Q.sum(1) + Q.sum(0)) + 0.5 * (X.sum(1) - X.sum(0))) / tot / ncell)
        return b


def unit_matrix(unit_of_group: np.ndarray, B: int, rng, units_subset=None) -> np.ndarray:
    """(B, n_groups) multiplicities: resample the units with replacement (units_subset: ids of the units that exist in this design), map to groups."""
    units = np.unique(unit_of_group) if units_subset is None else np.asarray(units_subset); U = len(units); pick = rng.integers(0, U, size=(B, U)); pos = {u: i for i, u in enumerate(units)}
    counts = np.zeros((B, U)); np.add.at(counts, (np.repeat(np.arange(B), U), pick.ravel()), 1)
    idx = np.array([pos.get(u, -1) for u in unit_of_group]); X = np.where(idx[None] >= 0, counts[:, np.clip(idx, 0, None)], 0.0); return X


def group_bootstrap(des: Design, unit_of_group: np.ndarray, B: int = 2000, seed: int = 0, chunk: int = 100) -> np.ndarray:
    """B bootstrap replicates of the estimate, resampling the units that occur in this design (a unit absent from every selected cell is not resampled)."""
    rng = np.random.default_rng(seed); present = np.unique(unit_of_group[np.unique(des.gidx[des.gidx >= 0])]); out = []
    for s in range(0, B, chunk):
        X = unit_matrix(unit_of_group, min(chunk, B - s), rng, present); out.append(np.nanmean(des.cell_values(X), axis=1))
    return np.concatenate(out)


def signflip_p(contrib: np.ndarray, unit_of_group: np.ndarray, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Two-sided randomisation p-value (random +-1 per unit applied to the unit sums of Design.perm_contributions(); see there) of the point estimate and the randomisation standard error sqrt(sum B_u^2)."""
    units = np.unique(unit_of_group); A = np.array([contrib[unit_of_group == u].sum() for u in units]); A = A[A != 0]; obs = abs(A.sum()); rng = np.random.default_rng(seed)
    if len(A) == 0: return 1.0, 0.0
    T = np.abs((rng.integers(0, 2, size=(n, len(A))) * 2 - 1) @ A); return float((1 + (T >= obs - 1e-15).sum()) / (1 + n)), float(np.sqrt((A ** 2).sum()))


def seed_level(des: Design, B: int = 4000, seed: int = 0) -> dict:
    """The earlier (seed-level) inference on the same cells: per-seed means, Wilcoxon over seeds, bootstrap over seeds."""
    v = des.point(); df = pd.DataFrame({"seed": des.cell_seed, "v": v}).dropna(); per = df.groupby("seed").v.mean(); rng = np.random.default_rng(seed)
    boots = rng.choice(per.to_numpy(), (B, len(per))).mean(1)
    try: p = float(wilcoxon(per).pvalue) if np.any(per.to_numpy() != 0) else 1.0
    except ValueError: p = 1.0
    return {"seed_n": len(per), "seed_lo": float(np.percentile(boots, 2.5)), "seed_hi": float(np.percentile(boots, 97.5)), "seed_wilcoxon_p": p, "seed_sd": float(per.std(ddof=1))}


def draw_bootstrap(des: Design, B: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample label draws within every (seed, split, budget) (needs >= 2 draws): isolates the label-draw component."""
    v = des.point(); df = pd.DataFrame({"seed": des.cell_seed, "split": des.cell_split, "budget": des.cell_budget, "draw": des.cell_draw, "v": v}).dropna(); rng = np.random.default_rng(seed)
    grp = [g.v.to_numpy() for _, g in df.groupby(["seed", "split", "budget"])]; return np.array([np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in grp])) for _ in range(B)])


def seed_resample(des: Design, B: int = 2000, seed: int = 0) -> np.ndarray:
    v = des.point(); df = pd.DataFrame({"seed": des.cell_seed, "v": v}).dropna(); per = df.groupby("seed").v.mean().to_numpy(); rng = np.random.default_rng(seed); return rng.choice(per, (B, len(per))).mean(1)


def ci(x, lo=2.5, hi=97.5): return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))

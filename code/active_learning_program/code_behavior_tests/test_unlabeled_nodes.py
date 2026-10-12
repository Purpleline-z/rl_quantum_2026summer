"""Unlabelled trajectory nodes: identity-safe (no near-duplicate of any label-sheet or ideal image), label-free, change the graph but not the candidate set."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint"
sys.path.insert(0, str(STUDY)); sys.path.insert(0, str(Path(__file__).parent))
import graph_typed as gt  # noqa: E402
import unlabeled_nodes as un  # noqa: E402
from test_judgment_unit import _judgments  # noqa: E402

pytestmark = pytest.mark.skipif(not un.load_unlabeled(), reason="representation cache not available")


def test_no_unlabelled_node_is_a_path_or_near_duplicate_of_a_sheet_or_ideal_image():
    extra = un.load_unlabeled(); man = pd.read_csv(un.REP / "manifest.csv"); man["key"] = man.path.str.replace(r"^.*/(original data/.*)$", r"\1", regex=True); x = np.load(un.REP / "cache/rheed_simclr_resnet18.npy").astype(np.float64)
    sheet = pd.read_csv(un.SHEET); used = set("original data/" + sheet.Image1_Path) | set("original data/" + sheet.Image2_Path)
    names = {n.removeprefix("unlabelled::") for n in extra}; assert names.isdisjoint(used) and all("Trajectories" in n for n in names)
    idx = {k: i for i, k in enumerate(man.key)}; z = x - x.mean(0); z /= np.linalg.norm(z, axis=1, keepdims=True)
    protected = [idx[k] for k in used if k in idx] + man.index[man.source == "ideal"].tolist()
    assert max(float((z[protected] @ z[idx[n]]).max()) for n in names) <= un.DUPLICATE_COSINE + 1e-9


def test_unlabelled_nodes_extend_the_graph_but_not_the_candidates():
    cands, labeled, cache = _judgments(); refs = {}
    base = gt.ImageGraph(cands, labeled, cache, refs); ext = gt.ImageGraph(cands, labeled, cache, refs, unlabeled=True)
    assert len(ext.paths) == len(base.paths) + len(un.load_unlabeled()) and ext.paths[:len(base.paths)] == base.paths
    assert ext.adjacency.shape[0] == len(ext.paths) and np.array_equal(ext.adjacency, ext.adjacency.T)

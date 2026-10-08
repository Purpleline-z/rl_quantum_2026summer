#!/usr/bin/env python3
"""Structure of the pairwise-preference data viewed as a graph: images are nodes, pair groups are edges."""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "active_learning_program"))
import run_frozen_encoder_task3 as harness  # noqa: E402


def components(edges):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges: parent[find(a)] = find(b)
    groups = Counter(find(x) for x in list(parent))
    return sorted(groups.values(), reverse=True)


def main() -> None:
    exp = harness.make_experiment(42, str(HERE.parents[2] / "data"), Path(tempfile.mkdtemp())); initial, candidates = exp.load_and_split()
    rows = []
    for pair_id, group in exp.groups.items():
        r = group.iloc[0]; rows.append({"pair_id": pair_id, "img1": r.resolved_img1, "img2": r.resolved_img2, "types": ",".join(sorted(set(group.canonical_type))),
                                       "rows": len(group), "winners": ",".join(sorted(group.Winner.astype(str)))})
    frame = pd.DataFrame(rows); edges = list(zip(frame.img1, frame.img2)); degree = Counter([x for e in edges for x in e])
    comps = components(edges)
    print(f"pair groups (edges): {len(frame)}; distinct images (nodes): {len(degree)}")
    print("degree distribution (images by number of pair groups):", dict(sorted(Counter(degree.values()).items())))
    print("connected components (sizes, nodes):", comps[:10], "total", len(comps))
    print("rows per pair group:", dict(sorted(Counter(frame.rows).items())))
    print("types per pair group:", dict(Counter(frame.types).most_common()))
    folders = Counter(Path(p).parent.parent.name + "/" + Path(p).parent.name for p in degree)
    print("images by trajectory folder:", dict(folders.most_common(6)))
    same_day = np.mean([Path(a).parent == Path(b).parent for a, b in edges]); print(f"fraction of pairs whose two images share a folder: {same_day:.2f}")
    outcome = Counter(w for ws in frame.winners for w in ws.split(","))
    print("row outcomes:", dict(outcome))
    frame.drop(columns=["img1", "img2"]).to_csv(HERE / "results" / "pair_graph_summary.csv", index=False)


if __name__ == "__main__":
    main()

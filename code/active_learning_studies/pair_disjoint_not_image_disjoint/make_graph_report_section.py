#!/usr/bin/env python3
"""Write report section 5.14 (graph-aware acquisition) from the judgment-unit cells; every number in the tables is computed here, none is typed by hand.

usage: make_graph_report_section.py            -> results/judgment_unit_study/graph_report_section.md
       make_graph_report_section.py --insert   -> additionally inserts/replaces the section in code/ACADEMIC_REPORT_DRAFT.md before '## 6. Conclusion'
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_graph_vs_random as agg

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "ACADEMIC_REPORT_DRAFT.md"
LABEL = {"graph_centrality_uncertainty": "Uncertainty x PageRank of the pair's images", "graph_bridge_uncertainty": "Uncertainty x boundary score",
         "graph_core_set": "Core-set on graph-propagated features", "uncertainty": "Uncertainty (no graph)",
         "graph_centrality_uncertainty_shuffled": "Uncertainty x PageRank, graph shuffled", "graph_bridge_uncertainty_shuffled": "Uncertainty x boundary score, graph shuffled",
         "graph_core_set_shuffled": "Core-set on propagated features, graph shuffled"}
ORDER = list(LABEL)
SPLIT = {"A": "A (10 initial / 20 val / 40 test groups)", "B": "B (classifier2-style 20% hold-out)"}


def fp(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def gains(split: str, condition: str, metric: str):
    cells = agg.load(split, condition); lower = agg.METRICS[metric][1]; sign = -1.0 if lower else 1.0
    per = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    random = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); paired = per.merge(random, on=["seed", "budget"]); paired["gain"] = sign * (paired[metric] - paired.random)
    seed_level = paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index()
    tests = {f: agg.wilcoxon_p(g.gain.to_numpy()) for f, g in seed_level.groupby("family")}; adjusted = agg.holm({k: v for k, v in tests.items() if not np.isnan(v)})
    mean = seed_level.groupby("family").gain.mean(); better = seed_level.groupby("family").gain.apply(lambda x: (x > 0).mean()); n = seed_level.groupby("family").seed.nunique()
    return mean, adjusted, better, n, seed_level


def main_table() -> str:
    lines = ["| Split | Condition | Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for split in ("A", "B"):
        for condition in ("single", "sequential"):
            ml, hl, bl, n, _ = gains(split, condition, "test_decisive_log_loss"); ma, ha, _, _, _ = gains(split, condition, "test_decisive_auc")
            for name in sorted(ORDER, key=lambda k: -ml[k]):
                lines.append(f"| {split} | {'single-shot' if condition == 'single' else 'sequential'} | {LABEL[name]} | {ml[name]:+.3f} | {hl[name]:.3f} | {ma[name]:+.3f} | {ha[name]:.3f} | {bl[name]:.0%} ({n[name]}) |")
    return "\n".join(lines)


def control_table() -> str:
    lines = ["| Split | Condition | Graph rule | log-loss: real − shuffled | p | AUC: real − shuffled | p |", "|---|---|---|---:|---:|---:|---:|"]
    for split in ("A", "B"):
        for condition in ("single", "sequential"):
            _, _, _, _, sl = gains(split, condition, "test_decisive_log_loss"); _, _, _, _, sa = gains(split, condition, "test_decisive_auc")
            wl = sl.pivot(index="seed", columns="family", values="gain"); wa = sa.pivot(index="seed", columns="family", values="gain")
            for real, fake in agg.CONTROLS.items():
                dl = (wl[real] - wl[fake]).dropna().to_numpy(); da = (wa[real] - wa[fake]).dropna().to_numpy()
                lines.append(f"| {split} | {'single-shot' if condition == 'single' else 'sequential'} | {LABEL[real]} | {dl.mean():+.3f} | {fp(agg.wilcoxon_p(dl))} | {da.mean():+.4f} | {fp(agg.wilcoxon_p(da))} |")
    return "\n".join(lines)


TEMPLATE = """### 5.14 Graph-aware acquisition on an image-similarity graph

**Question.** Pair groups share images and expert comparisons implicitly order the images, so a graph over images might carry information that the mean-pair embedding of the earlier strategies discards. We asked whether a graph-aware selection rule improves held-out preference prediction over Random and over the embedding-based rules.

**What the comparison graph looks like.** Taking images as nodes and the 168 usable pair groups as edges, the graph has 284 nodes and 117 connected components (largest 11 nodes); 237 images occur in exactly one pair group, 42 in two and 5 in three. Per reconstruction type the directed graph of decisive judgments (loser to winner) has 45–94 edges, connected components of at most 4–5 nodes, and only 0–6 images that both won and lost a comparison. The comparison graph is therefore nearly a matching, and a model that recovers a global ranking from it (such as GNNRank, He et al., ICML 2022) has no transitive structure to recover. (The working slides give 669 pairs and about 300 images; the data in this repository contain 638 judgment rows, 168 pair groups and 284 images.) The graph used below is instead built from image similarity.

**Graph and strategies.** At every selection step the graph has as nodes the images of the candidate judgments and of the judgments already revealed, and as edges the symmetrised ten nearest neighbours in the cached SimCLR feature space. It uses no labels and cannot contain a validation or test image, because a selector sees only these images. Three rules were implemented in `graph_strategies.py`: (i) own-head uncertainty multiplied by the percentile rank of the PageRank of the pair's two images, (ii) own-head uncertainty multiplied by a boundary score (the share of an image's neighbours that fall in another k-means cluster of eight), and (iii) farthest-first k-centre on [(a+b)/2, |a−b|, a·b] of features propagated twice over the graph (SGC). Each has a control in which the graph scores, or the propagated features, are permuted over the nodes. Reconstruction-type information (the ideal images as typed anchors) is not available to the selectors in this implementation, so the boundary score is an approximation of the cross-type boundary seen in the ideal images (112 of 770 five-nearest-neighbour edges among ideal images join different types, 52 of them HTR–RT13).

**Protocol.** The query unit is one (pair, type) judgment and the budget counts judgments (10, 20, 40, 60), as in the judgment-unit study; numbers are therefore not comparable with the group-budget tables of §5.12–5.13. Splits A and B, seeds, head schedule and endpoint are those of §5.12–5.13 (35 seeds, single-shot and sequential rounds of 10). The candidate pool holds 201–250 judgments in split A and 311–350 in split B. Gains are per-seed differences from the mean of five Random draws, averaged over budgets; the test is a Wilcoxon signed-rank test over seeds with Holm correction over the seven non-random rows of each block. There are eight blocks (two splits, two conditions, two metrics) and no correction across them.

**Diagnostic before the experiment.** As a check on whether graph smoothing changes what the features can predict, a linear Bradley–Terry head on frozen features was fit to the 264 decisive judgments (five folds by pair group, 20 repetitions). Raw features reach AUC 0.941; features smoothed over the similarity graph (1–4 steps, with or without a within-session temporal chain) reach 0.938–0.940, and a spectral embedding 0.909. This does not test a trained graph network or an acquisition rule.

{main}

**Results.** No graph rule is significantly better than Random after Holm correction in any block. The one significant gain in the table belongs to a control: the shuffled-graph core-set in split B sequential (log-loss +0.056, Holm p = 0.013), which shows that the coverage gain of that rule does not come from the graph. Uncertainty multiplied by PageRank is the only rule that is nominally positive in split A in both conditions (log-loss +0.034 single-shot and +0.048 sequential, raw p of about 0.012–0.015 but Holm p of 0.11 and 0.08; AUC +0.006 and +0.011); it is not positive in split B single-shot (−0.021) and its sequential gain there (+0.032, Holm p = 0.23) is matched by its shuffled control (+0.033). The boundary-score rule is never better than Random and is significantly worse in split B single-shot (log-loss −0.066, Holm p = 0.007; AUC −0.017, Holm p = 0.004); uncertainty without a graph is also below Random there (−0.047, Holm p = 0.055), so the boundary score does not repair the weakness of uncertainty sampling in that split. Core-set on propagated features does not differ from its shuffled control in either split.

Comparison with the shuffled controls (paired over seeds, unadjusted p-values, twelve comparisons per metric):

{control}

The PageRank rule is better than its shuffled control in split A (log-loss +0.064 single-shot, p = 0.037; +0.081 sequential, p < 0.001), but the shuffled control is itself worse than Random there (−0.030 and −0.033), so part of this difference reflects the control rather than the graph, and the difference is absent in split B.

**What this does and does not show.** Under these splits and with this encoder, selection on a ten-nearest-neighbour similarity graph is not better than Random or than the rules of §5.12–5.13, and the one nominally positive rule does not replicate across splits. The experiment does not test a trained graph network, a graph built from all 1,124 trajectory images (840 of them unlabelled), ideal images as typed anchors, or temporal edges within a session (the temporal constraints file marks these as tentative); the effect of an informative graph could be larger than that of these simple versions. The tests are low-powered for effects of the size seen elsewhere in this report (log-loss gains of 0.03–0.05 against a per-seed standard deviation of 0.09–0.14).

"""


def build() -> str:
    return TEMPLATE.replace("{main}", main_table()).replace("{control}", control_table())


if __name__ == "__main__":
    text = build(); out = HERE / "results" / "judgment_unit_study" / "graph_report_section.md"; out.write_text(text)
    if "--insert" in sys.argv:
        doc = REPORT.read_text(); doc = re.sub(r"### 5\.14 Graph-aware acquisition.*?(?=## 6\. Conclusion)", "", doc, flags=re.S)
        assert "## 6. Conclusion" in doc; REPORT.write_text(doc.replace("## 6. Conclusion", text + "## 6. Conclusion", 1))
    print(text)

#!/usr/bin/env python3
"""Report section 5.15 (confirmation of the frozen type-aware graph rules): tables and the criterion verdict are computed from the cells; the interpretation paragraph is read from
graph_exploration/confirm_narrative.md (written by hand after looking at the numbers; the file is optional).
usage: make_confirm_section.py [--insert]
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import aggregate_confirm as ac

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "ACADEMIC_REPORT_DRAFT.md"
NARRATIVE = HERE / "graph_exploration" / "confirm_narrative.md"

INTRO = """### 5.15 Confirmation of the type-aware graph acquisition rules on unseen seeds

**Design.** Three rules were frozen on development seeds 400-409 before any confirmation seed was run (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`; the freezing is recorded in `graph_exploration/AUTONOMOUS_RUN_LOG.md`). They were then run once on 25 seeds not used for design (410-429 and 42, 79, 123, 202, 303), under the protocol of §5.14 (query unit = one (pair, type) judgment; budgets 10, 20, 40 and 60 judgments; splits A and B; single-shot and sequential). Gains are per-seed differences from the mean of five Random draws, averaged over the four budgets; the p-value is a seed-level Wilcoxon signed-rank test, Holm-corrected over the three frozen candidates within each block; controls and embedding baselines are shown with raw p-values and are not part of the Holm family. The success criterion fixed in advance was: Holm p < 0.05 for log-loss in at least two of the four blocks including both splits, with a non-negative AUC gain in those blocks. The candidates were chosen because they looked best on the development seeds, whose per-seed standard deviation of the gain is 0.09-0.14, so development gains are optimistic estimates.

**Rules.** The graph rules build a ten-nearest-neighbour graph over the images of the candidate and revealed judgments and the typed reference images, spread the reference types over it (label spreading), predict the probability that a judgment is decisive from the resulting type-posterior features and the revealed outcomes, and select by farthest-first coverage in graph-propagated pair space weighted by that probability (and by own-head uncertainty for the primary rule); the two controls replace the decisive predictor by a type-only one or shuffle the posterior (§5.14 and `graph_exploration/DESIGN_AND_LITERATURE.md`).

"""


def build() -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer): ac.main(True)
    out = buffer.getvalue(); verdict, table = out.split("\n\n", 1) if "Success criterion" not in out.split("\n\n")[0] else (None, None)
    head, rest = out.split("Success criterion", 1); verdict_text = "Success criterion" + rest.split("| Split")[0]; table = "| Split" + rest.split("| Split", 1)[1]
    narrative = NARRATIVE.read_text() if NARRATIVE.exists() else "(Interpretation paragraph not yet written.)\n"
    replication = ""
    if (HERE / "graph_exploration" / "replication_narrative.md").exists():
        import aggregate_replication as rep
        replication = ("\n\n**Replication on fresh seeds (split B only).** Because the plain rule was significant in split B but its own controls had not been run, a replication was pre-registered in `graph_exploration/AUTONOMOUS_RUN_LOG.md` before it ran: seeds 430-459 (30 seeds never used before), split B, both conditions, with no change to the rule, plus its type-only and shuffled-posterior controls and the Core-set baseline. Gains are over Random as above; p-values are raw seed-level Wilcoxon tests (the paired differences are not corrected for multiplicity; H1 is corrected over the two conditions).\n\n"
                       + rep.markdown() + "\n" + (HERE / "graph_exploration" / "replication_narrative.md").read_text())
    return INTRO + table.strip() + "\n\n**Verdict against the pre-registered criterion.**\n\n```\n" + verdict_text.strip() + "\n```\n\n" + narrative + replication + "\n"


if __name__ == "__main__":
    text = build(); (HERE / "results" / "judgment_unit_study" / "confirm_report_section.md").write_text(text)
    if "--insert" in sys.argv:
        doc = REPORT.read_text(); doc = re.sub(r"### 5\.15 Confirmation of the type-aware.*?(?=## 6\. Conclusion)", "", doc, flags=re.S)
        assert "## 6. Conclusion" in doc; REPORT.write_text(doc.replace("## 6. Conclusion", text + "## 6. Conclusion", 1))
    print(text)

#!/usr/bin/env python3
"""Report section 5.20 (graph rules under the re-tuned head schedule; graph-regularised design): tables from the cells (make_overnight2_tables.py), prose from
graph_exploration/overnight2_narrative.md (hand-written after looking at the numbers).  usage: make_overnight2_section.py [--insert]"""
import re
import sys
from pathlib import Path

import make_overnight2_tables as mt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "ACADEMIC_REPORT_DRAFT.md"
NARR = HERE / "graph_exploration" / "overnight2_narrative.md"


def build() -> str:
    text = NARR.read_text() if NARR.exists() else "(narrative not yet written)\n"
    parts = (HERE / "graph_exploration" / "overnight2_parts56.md"); text = text.replace("{PARTS_5_6}", parts.read_text() if parts.exists() else "")
    typed = (HERE / "graph_exploration" / "overnight2_part_typed.md"); text = text.replace("{PART5_TYPED}", typed.read_text() if typed.exists() else "")
    text = text.replace("{TABLE_P5}", mt.table("P5")).replace("{POOLED_TYPED}", (HERE / "results" / "new_methods" / "pooled_typed_report.txt").read_text().rstrip())
    text = text.replace("{TABLE_P1}", mt.table("P1")).replace("{TABLE_P3}", mt.table("P3")).replace("{TABLE_P4}", mt.table("P4")).replace("{TABLE_P5}", mt.table("P5")).replace("{TABLE_P6}", mt.table("P6")).replace("{TABLE_P2}", (HERE / "results" / "new_methods" / "graph_p2" / "choose_p2_summary.txt").read_text().rstrip())
    return text.rstrip("\n") + "\n\n"


if __name__ == "__main__":
    section = build(); (HERE / "results" / "new_methods" / "section_5_20.md").write_text(section)
    if "--insert" in sys.argv:
        doc = REPORT.read_text(); doc = re.sub(r"### 5\.20 .*?(?=## 6\. Conclusion)", "", doc, flags=re.S)
        assert doc.count("## 6. Conclusion") == 1; REPORT.write_text(doc.replace("## 6. Conclusion", section + "## 6. Conclusion", 1))
    print(section[:600])

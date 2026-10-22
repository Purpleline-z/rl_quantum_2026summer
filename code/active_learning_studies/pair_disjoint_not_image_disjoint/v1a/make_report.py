#!/usr/bin/env python3
"""Fill REPORT_V1a.md from report_template.md using results/tables.md, results/tuning_grid.csv and the run logs (no number is typed by hand)."""
import re
from pathlib import Path

import pandas as pd

import aggregate as A

HERE = Path(__file__).resolve().parent; frame = pd.read_csv(HERE / "results" / "all_rows.csv")
seconds = [int(x) for f in (HERE / "logs").glob("run_*.log") for x in re.findall(r"rows in (\d+)s", f.read_text())]
text = (HERE / "report_template.md").read_text()
text = text.replace("%%SEED_SECONDS%%", f"{sum(seconds) / len(seconds):.0f}").replace("%%CPU_H%%", f"{sum(seconds) / 3600:.1f}").replace("%%ENS%%", f"{frame[frame.rule == 'ensemble_bald'].rule_seconds.mean():.0f}")
text = text.replace("%%TUNING%%", A.md(pd.read_csv(HERE / "results" / "tuning_grid.csv"))).replace("%%TABLES%%", (HERE / "results" / "tables.md").read_text())
(HERE / "REPORT_V1a.md").write_text(text); print("written", len(text))

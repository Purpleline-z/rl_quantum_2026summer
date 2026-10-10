"""Markdown tables of the pre-registered tests, generated from the confirmatory result files (nothing typed by hand).  Usage: python3 nd_confirm_markdown.py prereg1_spec.json results/confirm1 > results/CONFIRM1_TABLE.md"""
import json, sys
from pathlib import Path
import pandas as pd
import nd_confirm as c

spec = json.loads(Path(sys.argv[1]).read_text()); out = c.run(spec, c.load(sys.argv[2:]))
lines = ["| family | split | metric | cell A - cell B | seeds | gain [95% interval] | seeds better | p | Holm |", "|---|---|---|---|---|---|---|---|---|"]
fmt = lambda p: "< 0.0001" if p < 1e-4 else f"{p:.4f}"
for _, r in out.iterrows():
    lines.append(f"| {r.family} | {r.split} | {r.metric} | {r.a[0]}:{r.a[1]} - {r.b[0]}:{r.b[1]} | {r.n_seeds} | {r.gain:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | {r.share_better:.0%} | {fmt(r.p)} | {fmt(r.holm)} |")
print("\n".join(lines))

#!/usr/bin/env python3
"""Descriptive facts about the calendar axis of the 168 pair groups and the six cuts (writes results/td/data_facts.md; run: python3 td_data_facts.py)."""
import collections, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_directions"))
import numpy as np
import nd_core as core, td_core as td, rt_core as rt

def main():
    cache = core.feature_cache(); lines = []
    with tempfile.TemporaryDirectory() as s:
        ctx = core.make_ctx(10500, "B", s, cache); order = td.calendar_order(ctx)
        assign = [td.group_time(ctx, g) for g in order]; run_of_group = collections.Counter(t[0] for t in assign)
        cross = sum(1 for g in order if len({rt.run_of(p) for p in td.group_images(ctx, g)}) == 2)
        names = {0: "RR220204A (2022-02-04)", 1: "RR220206A (2022-02-06)", 2: "RR220411A (2022-04-11)"}
        lines += [f"- groups: {len(order)}; groups that can be assigned a time: {len(order)} (every image of every group is a trajectory frame with a parsable run and frame number); groups that cannot: 0",
                  f"- cross-run groups: {cross} of {len(order)} (their time is that of the later image, calendar rank of the run first); same-run groups: {len(order) - cross}",
                  "- groups by run of the group time: " + ", ".join(f"{names[r]}: {run_of_group[r]}" for r in sorted(run_of_group)),
                  "", "| cut (n past groups) | seed 10500 jitter: n past | past groups kept (identity rule) | past judgments | FA test judgments (decisive) | FB test judgments (decisive) | nodes ind | nodes trans_test | nodes trans_fut |", "|---|---|---|---|---|---|---|---|---|"]
        for cut in td.CUTS:
            ctx = td.make_forward_ctx(10500, cut, s, cache); t = td.forward_tests(ctx); sets = td.exclusion_sets(ctx); n = {m: len(td.td_graph(ctx, m).paths) for m in ("ind", "trans_test", "trans_fut")}
            lines.append(f"| {cut} | {ctx.forward['n_train']} | {ctx.forward['n_train_kept']} | {len(ctx.initial) + len(ctx.pool)} | {len(t['FA'].k)} | {len(t['FB'].k)} | {n['ind']} | {n['trans_test']} | {n['trans_fut']} |")
    out = Path(core.HERE) / "results" / "td" / "data_facts.md"; out.write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__": main()

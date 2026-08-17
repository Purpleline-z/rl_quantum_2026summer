# Legacy Sequential Exploration

## What was run

This folder preserves an early sequential active-learning exploration: 70
initial labeled pair groups, a 100-pair candidate pool, acquisition in batches
of two, and a 30-image ideal-image test set. Sequential selection means the
model is retrained and the next ranking can change after every small batch. That
is a different scientific object from the later fixed-budget curves, where each
budget is evaluated from a declared initial set under a fixed selector protocol.

## What the artifacts mean

- **Observed artifact:** the files in this folder show how selector rankings and
  downstream ideal-image accuracy behaved during a historical, sequential run.
- **What it means:** they are useful for spotting whether a selector changes its
  behavior as labels accumulate and for generating hypotheses about coverage or
  uncertainty. They do not estimate the current selector effect because their
  split, test size, acquisition schedule, and likely repeated-image exposure
  differ from the current pair-disjoint benchmark.
- **Research decision:** preserve these outputs as origin evidence for later
  budget-curve work. Test any hypothesis they suggest using the current
  five-seed, budget-aware protocol rather than combining their numbers with its
  summary table.

## Where to look next

For the current protocol and evidence map, read the
[`pair-disjoint study README`](../../../README.md), the
[`results guide`](../../README.md), and the root
[`technical report`](../../../../../TECHNICAL_REPORT.md).

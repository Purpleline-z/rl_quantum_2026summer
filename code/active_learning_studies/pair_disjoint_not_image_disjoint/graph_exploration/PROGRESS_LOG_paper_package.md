# Progress log — paper package (branch `claude/paper-package-20261013`)

| stage | what | commit |
|---|---|---|
| 0 | read COMMON_RULES, result documents, code; fetched branches; cut the branch from `claude/redteam-20261013`; imported result trees and documents of `claude/graph-mechanism-20261013` and `claude/graph-endpoint-1013` unchanged | b83fcb1bc |
| 1 | tables (confirmations recomputed with each study's own test code, ladder, curves, selection, dose, strata), figures 1–6 | 8d72f6dd4 |
| 2 | claims table, prose with number checker (abstract, Chinese summary, assumptions and limits), Makefile, README, tests, handoff, GPU tasks, pooled estimates | 26503a5c2, 01fa62a4b |
| 3 | red-team confirmation block merged (its per-judgment files became complete): strata, claims, figure 6 and prose updated; fresh-clone reproduction (`make all` byte-identical) | 0453c2846 |
| 4 | final sync of the red team's files and final push | see `git log` |

Decisions worth knowing: (1) the package runs nothing and uses no seed block; (2) the anchor-far stratum uses the brief's threshold 0.03 (all typed anchors), the red team's frozen threshold 0.01 is shown as sensitivity and the verdict is shown to depend on it; (3) statuses are recomputed from each study's frozen rule, not copied from documents; (4) results imported from other branches are unchanged copies (`sync_branches.sh` refreshes them).

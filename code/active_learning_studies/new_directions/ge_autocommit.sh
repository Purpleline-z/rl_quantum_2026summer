#!/bin/bash
# commits/pushes new results/ge cells every 4 minutes while GE jobs run (stop hook wants a clean tree)
cd "$(git rev-parse --show-toplevel)" || exit 1; BRANCH=$(git rev-parse --abbrev-ref HEAD)
commit() { git add code/active_learning_studies/new_directions/results/ge 2>/dev/null; git diff --cached --quiet || { git commit -q -m "graph-endpoint: result cells (auto)" && git push -q origin "$BRANCH"; }; }
while pgrep -f "ge_run_|ge_dev|ge_conf" >/dev/null; do commit; sleep 240; done
commit

#!/bin/bash
cd "$(git rev-parse --show-toplevel)" || exit 1; BRANCH=$(git rev-parse --abbrev-ref HEAD)
commit() { git add code/active_learning_studies/new_directions/results/ae 2>/dev/null; git diff --cached --quiet || { git commit -q -m "anchor-exchange: result cells (auto)" && git push -q origin "$BRANCH"; }; }
while pgrep -f "ae_run|ae_dev|ae_conf" >/dev/null; do commit; sleep 240; done
commit

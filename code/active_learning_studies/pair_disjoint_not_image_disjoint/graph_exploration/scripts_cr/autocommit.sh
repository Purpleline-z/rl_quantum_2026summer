#!/bin/bash
# commit + push result cells of the CR study every 10 minutes (specific paths only)
cd "$(dirname "$0")/../../../../.."
while true; do
  sleep 600
  git add code/active_learning_studies/new_directions/results/cr 2>/dev/null
  git commit -q -m "cross-run: result cells (auto)" -- code/active_learning_studies/new_directions/results/cr 2>/dev/null && git push -q origin HEAD:claude/cross-run-20261014 2>/dev/null
done

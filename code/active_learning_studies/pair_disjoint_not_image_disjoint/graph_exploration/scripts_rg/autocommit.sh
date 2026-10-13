#!/bin/bash
# commit + push result cells of the G2 study every 10 minutes (specific paths only)
cd "$(dirname "$0")/../../../../.."
while true; do
  sleep 600
  git add code/active_learning_studies/new_directions/results/rg 2>/dev/null
  git commit -q -m "regularisers: result cells (auto)" -- code/active_learning_studies/new_directions/results/rg 2>/dev/null && git push -q origin HEAD:claude/regularisers-20261013 2>/dev/null
done

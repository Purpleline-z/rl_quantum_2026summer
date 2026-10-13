#!/bin/bash
# commits audit result cells every 8 minutes (specific paths only); stop with: touch /tmp/rt_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/rt_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/rt code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*redteam* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "redteam: audit result cells (auto)" && git push -q origin HEAD 2>/dev/null; }
  sleep 480
done

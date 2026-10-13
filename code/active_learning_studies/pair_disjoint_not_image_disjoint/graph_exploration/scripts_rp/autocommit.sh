#!/bin/bash
# commits result cells every 8 minutes (specific paths only); stop with: touch /tmp/rp_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/rp_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/rp code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*representation* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "representation: result cells (auto)" && git push -q origin HEAD 2>/dev/null; }
  sleep 480
done

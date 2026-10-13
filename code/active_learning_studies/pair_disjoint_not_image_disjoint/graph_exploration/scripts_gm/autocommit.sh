#!/bin/bash
# commits result cells every 8 minutes (specific paths only); stop with: touch /tmp/gm_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/gm_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/gm code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*graph_mechanism* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "graph-mechanism: result cells (auto)" && git push -q origin HEAD 2>/dev/null; }
  sleep 480
done

#!/bin/bash
# commits result cells every 8 minutes (specific paths only); stop with: touch /tmp/gl_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/gl_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/gl code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*graph_learner* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "graph-learner: result cells (auto)" && git push -q origin HEAD 2>/dev/null; }
  sleep 480
done

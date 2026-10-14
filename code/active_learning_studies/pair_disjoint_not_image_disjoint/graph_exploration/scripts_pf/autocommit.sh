#!/bin/bash
# commits result cells every 10 minutes (specific paths only); stop with: touch /tmp/pf_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/pf_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/pf code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*physics* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "physics-features: result cells (auto)" && git push -q origin HEAD 2>/dev/null; }
  sleep 600
done

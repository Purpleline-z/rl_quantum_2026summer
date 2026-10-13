#!/bin/bash
# commits result cells every 8 minutes (specific paths only); stop with: touch /tmp/sh_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/sh_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/sh code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*selection_htr* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "selection-htr: result cells (auto)" && git push -q origin HEAD 2>/dev/null; }
  sleep 480
done

#!/bin/bash
# commits result cells and documents of the transductive study every 8 minutes (specific paths only); stop with: touch /tmp/td_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/td_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/td code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*transductive* code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*TD* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "transductive: result cells (auto)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:claude/transductive-20261014 2>/dev/null; }
  sleep 480
done

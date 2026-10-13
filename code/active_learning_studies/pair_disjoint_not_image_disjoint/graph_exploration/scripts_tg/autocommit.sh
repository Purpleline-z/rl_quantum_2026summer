#!/bin/bash
# commits result cells and docs of the temporal-graph study every 8 minutes (specific paths only); stop with: touch /tmp/tg_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/tg_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/tg code/active_learning_studies/pair_disjoint_not_image_disjoint/graph_exploration/*temporal* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "temporal-graph: result cells (auto)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:claude/temporal-graph-20261013 2>/dev/null; }
  sleep 480
done

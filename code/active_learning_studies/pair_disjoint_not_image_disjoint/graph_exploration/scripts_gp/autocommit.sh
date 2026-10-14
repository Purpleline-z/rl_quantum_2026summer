#!/bin/bash
# commits and pushes only the graph-GP confirmation cells (every 10 minutes) until logs_gp/conf.done exists
ROOT=/home/user/rl_quantum_2026summer; P=code/active_learning_studies/new_directions/results/gp
while true; do
  cd $ROOT
  for d in conf1 conf1_blk conf1_cold conf2 conf2_blk; do [ -d $P/$d ] && git add $P/$d; done
  if ! git diff --cached --quiet -- $P 2>/dev/null; then git commit -q -m "graph-gp: confirmation result cells (auto)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CBZ8AGJmxLBPXF6gn6Bthq" -- $P && git push -q origin claude/graph-gp-20261014; fi
  [ -f $ROOT/code/active_learning_studies/new_directions/logs_gp/conf.done ] && break
  sleep 600
done

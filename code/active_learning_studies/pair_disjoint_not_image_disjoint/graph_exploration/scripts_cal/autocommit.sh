#!/bin/bash
# commits and pushes only the calibration result cells (every 10 minutes) until logs_cal/conf1.done exists
cd "$(dirname "$0")/../../../new_directions"; P=code/active_learning_studies/new_directions/results/cal
while true; do
  cd /home/user/rl_quantum_2026summer
  for d in conf1 conf1_aw8; do [ -d $P/$d ] && git add $P/$d; done
  if ! git diff --cached --quiet -- $P 2>/dev/null; then git commit -q -m "calibration: confirmation result cells (auto)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CBZ8AGJmxLBPXF6gn6Bthq" -- $P && git push -q origin claude/calibration-20261013; fi
  [ -f code/active_learning_studies/new_directions/logs_cal/conf1.done ] && break
  sleep 600
done

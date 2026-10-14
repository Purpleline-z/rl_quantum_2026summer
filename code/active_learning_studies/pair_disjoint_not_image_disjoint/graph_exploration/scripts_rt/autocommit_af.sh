#!/bin/bash
# commits anchor-far result cells every 8 minutes (specific paths only); stop with: touch /tmp/af_stop_autocommit
cd "$(git rev-parse --show-toplevel)"
while [ ! -f /tmp/af_stop_autocommit ]; do
  git add code/active_learning_studies/new_directions/results/rt/af_* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "anchor-far: confirmation result cells (auto)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0193nsACmxbVcEDQR2yut9j1" && git push -q origin HEAD 2>/dev/null; }
  sleep 480
done

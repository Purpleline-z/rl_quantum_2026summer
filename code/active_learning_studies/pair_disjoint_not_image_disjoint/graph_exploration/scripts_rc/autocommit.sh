#!/bin/bash
# commit + push result cells every 10 minutes (only result and log paths); usage: autocommit.sh BRANCH
cd "$(dirname "$0")/../../../../.."; B=${1:-claude/recipe-20261013}
while true; do
  sleep 600
  git add code/active_learning_studies/new_directions/results/rc 2>/dev/null
  git commit -q -m "recipe: result cells (auto)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01BQsBiG3qeBpmBZ2TCHK46Y" 2>/dev/null && git push -q origin HEAD:$B 2>/dev/null
done

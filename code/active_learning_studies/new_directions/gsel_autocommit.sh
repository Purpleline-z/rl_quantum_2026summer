#!/bin/bash
# Commits and pushes new gs_* result cells every 4 minutes while gsel jobs run (stop hook wants a clean tree). Adds only results/gs_* and logs-free paths.
cd "$(git rev-parse --show-toplevel)" || exit 1; BRANCH=$(git rev-parse --abbrev-ref HEAD)
commit() {
  git add code/active_learning_studies/new_directions/results/gs_* 2>/dev/null
  git diff --cached --quiet || { git commit -q -m "gsel result cells (in progress)" && git push -q origin "$BRANCH"; }
}
while pgrep -f "gsel_select.py|gsel_dev|gsel_conf" >/dev/null; do commit; sleep 240; done
commit

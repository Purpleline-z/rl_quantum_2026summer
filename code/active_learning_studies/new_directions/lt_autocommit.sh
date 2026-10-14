#!/bin/bash
cd "$(git rev-parse --show-toplevel)" || exit 1; BRANCH=$(git rev-parse --abbrev-ref HEAD)
commit() { git add code/active_learning_studies/new_directions/results/lt code/active_learning_studies/new_directions/logs_lt 2>/dev/null; git diff --cached --quiet || { git commit -q -m "latent-type: result cells (auto)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GoF52iqt5GMwMLYv6VT66j" && git push -q origin "$BRANCH"; }; }
while pgrep -f "lt_run|lt_diag|rt_a1b|lt_chain" >/dev/null; do commit; sleep 300; done
commit

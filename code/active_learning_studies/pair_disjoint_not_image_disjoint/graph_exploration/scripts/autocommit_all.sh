#!/bin/bash
# Commits and pushes new result cells every 4 minutes while an experiment process is alive (the repo's stop hook demands a clean tree). Only the results folder is added.
cd "$(git rev-parse --show-toplevel)" || exit 1; BRANCH=$(git rev-parse --abbrev-ref HEAD)
commit() {
  git add code/active_learning_studies/pair_disjoint_not_image_disjoint/results/judgment_unit_study code/active_learning_studies/pair_disjoint_not_image_disjoint/results/new_methods
  git diff --cached --quiet || { git commit -q -m "Result cells (in progress)" && git push -q origin "$BRANCH"; }
}
while pgrep -f "judgment_unit_study.py|new_methods_study.py|graph_methods_study.py" >/dev/null; do commit; sleep 240; done
commit

#!/usr/bin/env bash
# Refreshes the stored results of the studies from their branches (only files that are new or newer; nothing is deleted or edited).  Run from anywhere inside the repository, then `make all`.
# The package itself never calls git except in pp_provenance.py (read-only history queries).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch origin
S=code/active_learning_studies; G=$S/pair_disjoint_not_image_disjoint/graph_exploration
checkout() { git checkout "origin/claude/$1" -- "${@:2}"; }
checkout redteam-20261013          $S/new_directions/results/rt $G/PREREGISTRATION_RT1.md $G/results_narrative_redteam.md $G/rt_tables.py $G/rt_confirm.py $G/rt_report.py
checkout graph-mechanism-20261013  $S/new_directions/results/gm $G/RESULTS_graph_mechanism.md
checkout graph-endpoint-1013       $S/new_directions/results/ge $G/RESULTS_graph_endpoint.md
checkout selection-htr-20261013    $S/new_directions/results/sh $G/RESULTS_selection_htr.md
echo "synced; now: cd $G/PAPER_PACKAGE && make all   (update the tip commits in pp_registry.py)"

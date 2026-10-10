#!/bin/bash
# usage: status.sh P1|P2|P3 TARGET_PER_BLOCK  -- cell counts per block and process count; exit 0 when every block has TARGET cells
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
case "$1" in P1) OUT=graph_p1;; P2) OUT=graph_p2;; P3) OUT=graph_p3;; esac
done=1; line="$(date -u +%H:%M) up $(uptime -p | cut -c1-12) procs $(ps -eo cmd | grep -c '[g]raph_methods_study.py')"
for S in A B; do for m in single sequential; do n=$(ls results/new_methods/$OUT/$S/$m 2>/dev/null | grep -c '^seed.*\.json$'); line="$line | $S-$m $n"; [ "$n" -ge "$2" ] || done=0; done; done
echo "$line"; [ $done = 1 ]

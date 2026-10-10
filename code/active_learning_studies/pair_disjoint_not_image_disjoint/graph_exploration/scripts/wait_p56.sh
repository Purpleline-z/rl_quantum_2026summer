#!/bin/bash
# waits up to ~9.5 min for P5/P6 to finish; relaunches dead processes; prints one status line
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
timeout 575 bash -c 'for i in $(seq 1 28); do [ $(ps -eo cmd | grep -c "[g]raph_methods_study.py") -eq 0 ] && break; read -t 20 <> <(:) || true; done'
cnt() { ls results/new_methods/$1/$2/$3 2>/dev/null | grep -c '^seed.*json$'; }
p5=1; p6=1
for S in A B; do [ $(cnt graph_p5 $S single) -ge 400 ] && [ $(cnt graph_p5 $S sequential) -ge 360 ] || p5=0; [ $(cnt graph_p6 $S single) -ge 360 ] && [ $(cnt graph_p6 $S sequential) -ge 320 ] || p6=0; done
N=$(ps -eo cmd | grep -c "[g]raph_methods_study.py")
echo "$(date -u +%H:%M) $(uptime -p | cut -c1-14) procs $N | P5 A $(cnt graph_p5 A single)/$(cnt graph_p5 A sequential) B $(cnt graph_p5 B single)/$(cnt graph_p5 B sequential) of 400/360 | P6 A $(cnt graph_p6 A single)/$(cnt graph_p6 A sequential) B $(cnt graph_p6 B single)/$(cnt graph_p6 B sequential) of 360/320 | done P5=$p5 P6=$p6"
if [ "$N" = "0" ]; then [ $p5 = 0 ] && graph_exploration/scripts/phase_run.sh P5; [ $p6 = 0 ] && graph_exploration/scripts/phase_run.sh P6; echo relaunched-if-needed; fi

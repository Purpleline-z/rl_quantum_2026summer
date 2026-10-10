#!/bin/bash
# usage: status_dir.sh OUTDIR TARGET_SINGLE TARGET_SEQ  -- like status.sh but for any results/new_methods/<OUTDIR> and separate targets
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$HERE" || exit 1
done=1; line="$(date -u +%H:%M) procs $(ps -eo cmd | grep -c '[g]raph_methods_study.py')"
for S in A B; do for m in single sequential; do n=$(ls results/new_methods/$1/$S/$m 2>/dev/null | grep -c '^seed.*\.json$'); t=$2; [ $m = sequential ] && t=$3; line="$line | $S-$m $n/$t"; [ "$n" -ge "$t" ] || done=0; done; done
echo "$line"; [ $done = 1 ]

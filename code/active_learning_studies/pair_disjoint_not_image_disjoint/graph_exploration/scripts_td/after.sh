#!/bin/bash
# usage: after.sh "<pattern in the cmdline of the running jobs>" <run_td.sh args...> : waits until no td_run.py process whose cmdline contains the pattern remains, then runs run_td.sh
pat="$1"; shift
while pgrep -f "^python3 td_run.py.*${pat}" > /dev/null; do sleep 20; done
bash "$(dirname "$0")/run_td.sh" "$@"

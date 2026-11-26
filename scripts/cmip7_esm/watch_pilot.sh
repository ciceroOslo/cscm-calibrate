#!/bin/bash
# Emit one line per finished fit, per log with an error, and when the runner exits.
#   watch_pilot.sh [logdir] [runs-log name] [runner script name] [lines already seen]
L=${1:-/scratch/bensan/logs/cmip7_esm}
R=${2:-pilot.log}
S=${3:-run_pilot.sh}
n=${4:-0}
seen=""
while true; do
  c=$(cat $L/$R 2>/dev/null | wc -l)
  if [ "$c" -gt "$n" ]; then tail -n $((c - n)) $L/$R; n=$c; fi
  for f in $(grep -l "Traceback\|Error" $L/*.log 2>/dev/null); do
    case " $seen " in *" $f "*) ;; *) echo "ERROR in $f"; seen="$seen $f";; esac
  done
  grep -q "all done" $L/$R 2>/dev/null && break
  [ "$(pgrep -fc $S)" = "0" ] && { echo "runner exited"; break; }
  sleep 30
done

#!/bin/bash
# Pilot: concentration-driven joint fits (full + leave-one-tier-out) for two CMIP7 ESMs.
# Two fits at a time x 24 workers = 48 cores.
cd "$(dirname "$0")/../.."
PY="/storage/no-backup-nac/users/bensan/cscm-calibrate-esm-perfect-model/.venv-scm/bin/python -W ignore scripts/cmip7_esm/fit_esm.py"
L=/scratch/bensan/logs/cmip7_esm
mkdir -p $L
job() {  # source holdout
  h=${2:+--holdout $2}
  tag=$1_${2:-full}
  { $PY fit $1 $h --workers 24 --de-iter 60 --nsteps 800 --nwalkers 40 &&
    $PY predict $1 $h --workers 24 --ndraw 300; } 2>&1 | grep -v "^2100$" > $L/$tag.log
  echo "$(date) $tag done" >> $L/pilot.log
}
export -f job; export PY L
printf "%s\n" "UKESM1-3-LL" "CanESM6-0-MR" "UKESM1-3-LL H" "UKESM1-3-LL VL" \
  "CanESM6-0-MR H" "CanESM6-0-MR M" "CanESM6-0-MR VL" |
  xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) pilot all done" >> $L/pilot.log

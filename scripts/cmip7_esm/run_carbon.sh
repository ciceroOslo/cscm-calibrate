#!/bin/bash
# Stage 2 (emissions-driven, carbon cycle) on top of the stage-1 +eps full fits.
# Full + leave-one-tier-out per ESM; 2 fits x 24 workers = 48 cores.
cd "$(dirname "$0")/../.."
PY="/storage/no-backup-nac/users/bensan/cscm-calibrate-esm-perfect-model/.venv-scm/bin/python -W ignore scripts/cmip7_esm/fit_carbon.py"
L=/scratch/bensan/logs/cmip7_esm_carbon
mkdir -p $L
job() {  # source [holdout]
  h=${2:+--holdout $2}
  tag=$1_${2:-full}
  { $PY fit $1 --variant eps $h --workers 24 && $PY predict $1 --variant eps $h --workers 24; } 2>&1 |
    grep -v "^2100$" > $L/$tag.log
  echo "$(date) $tag done" >> $L/runs.log
}
export -f job; export PY L
printf "%s\n" "UKESM1-3-LL" "CanESM6-0-MR" "UKESM1-3-LL H" "CanESM6-0-MR H" "UKESM1-3-LL M" "CanESM6-0-MR M" \
  "UKESM1-3-LL VL" "CanESM6-0-MR VL" | xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) carbon all done" >> $L/runs.log

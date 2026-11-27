#!/bin/bash
# v4 carbon stage (ocean-flux offset, land-use CO2 scaling, forcing scalings re-estimated on
# emissions-driven runs, longer chains) on top of the v3 stage-1 fits; plus the remaining
# v3 stage-1 leave-one-tier-out fits. Waits for any running v3 stage-1 jobs first.
# 2 fits x 24 workers = 48 cores.
cd "$(dirname "$0")/../.."
PY=/storage/no-backup-nac/users/bensan/cscm-calibrate-esm-perfect-model/.venv-scm/bin/python
L=/scratch/bensan/logs/cmip7_esm_v3
job() {  # stage source [holdout]
  h=${3:+--holdout $3}
  tag=$1_$2_${3:-full}
  s=$([ $1 = carbon ] && echo fit_carbon.py || echo fit_esm.py)
  { $PY -W ignore scripts/cmip7_esm/$s fit $2 --variant eps $h --workers 24 &&
    $PY -W ignore scripts/cmip7_esm/$s predict $2 --variant eps $h --workers 24; } 2>&1 |
    grep -v "^2100$" > $L/$tag.log
  echo "$(date) $tag done" >> $L/runs.log
}
export -f job; export PY L
while pgrep -f "job stage1" > /dev/null; do sleep 60; done
echo "$(date) v4 start" >> $L/runs.log
printf "%s\n" "carbon UKESM1-3-LL" "carbon CanESM6-0-MR" | xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) v4 carbon full done" >> $L/runs.log
printf "%s\n" "carbon UKESM1-3-LL VL" "carbon CanESM6-0-MR VL" "stage1 UKESM1-3-LL VL" "stage1 CanESM6-0-MR VL" \
  "carbon UKESM1-3-LL H" "carbon CanESM6-0-MR H" "stage1 CanESM6-0-MR M" "carbon UKESM1-3-LL M" \
  "carbon CanESM6-0-MR M" | xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) v4 all done" >> $L/runs.log

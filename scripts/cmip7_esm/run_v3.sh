#!/bin/bash
# v3: 1850 piControl-consistent start, RCMIP3 volcanic (ciceroscm was silently using its IPCC
# default), free land/ocean carbon response functions. +eps only.
# Stage-1 full fits, then carbon full fits, then all leave-one-tier-out fits.
# 2 fits x 24 workers = 48 cores.
cd "$(dirname "$0")/../.."
PY=/storage/no-backup-nac/users/bensan/cscm-calibrate-esm-perfect-model/.venv-scm/bin/python
L=/scratch/bensan/logs/cmip7_esm_v3
mkdir -p $L
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
printf "%s\n" "stage1 UKESM1-3-LL" "stage1 CanESM6-0-MR" | xargs -P 2 -I{} bash -c 'job {}'
printf "%s\n" "carbon UKESM1-3-LL" "carbon CanESM6-0-MR" | xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) full fits done" >> $L/runs.log
printf "%s\n" "stage1 UKESM1-3-LL H" "stage1 CanESM6-0-MR H" "stage1 UKESM1-3-LL VL" "stage1 CanESM6-0-MR VL" \
  "stage1 CanESM6-0-MR M" "carbon UKESM1-3-LL VL" "carbon CanESM6-0-MR VL" "carbon UKESM1-3-LL H" \
  "carbon CanESM6-0-MR H" "carbon UKESM1-3-LL M" "carbon CanESM6-0-MR M" | xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) v3 all done" >> $L/runs.log

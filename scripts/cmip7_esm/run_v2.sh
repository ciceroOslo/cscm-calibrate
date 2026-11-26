#!/bin/bash
# v2: persistent-discrepancy likelihood, widened bounds; +eps and +eps with state-dependent
# feedback; full + leave-one-tier-out for the two pilot ESMs. 2 fits x 24 workers = 48 cores.
cd "$(dirname "$0")/../.."
PY="/storage/no-backup-nac/users/bensan/cscm-calibrate-esm-perfect-model/.venv-scm/bin/python -W ignore scripts/cmip7_esm/fit_esm.py"
L=/scratch/bensan/logs/cmip7_esm_v2
mkdir -p $L
job() {  # variant source [holdout]
  h=${3:+--holdout $3}
  tag=$1_$2_${3:-full}
  { $PY fit $2 --variant $1 $h --workers 24 && $PY predict $2 --variant $1 $h --workers 24; } 2>&1 |
    grep -v "^2100$" > $L/$tag.log
  echo "$(date) $tag done" >> $L/runs.log
}
export -f job; export PY L
{
  for v in eps eps_state; do echo "$v UKESM1-3-LL"; echo "$v CanESM6-0-MR"; done
  for v in eps eps_state; do
    for h in H VL; do echo "$v UKESM1-3-LL $h"; done
    for h in H M VL; do echo "$v CanESM6-0-MR $h"; done
  done
} | xargs -P 2 -I{} bash -c 'job {}'
echo "$(date) v2 all done" >> $L/runs.log

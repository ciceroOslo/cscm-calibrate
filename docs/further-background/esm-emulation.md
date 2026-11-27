# Calibration to individual ESMs

CICERO-SCM is calibrated separately to each CMIP Earth system model (ESM),
using all of that model's output: history, every ScenarioMIP tier, and the
idealised abrupt-4xCO2 and 1pctCO2 runs. The per-ESM calibrations serve two
purposes:

1. emulating each ESM in the scenarios it has not run (filling the sparse
   CMIP7 scenario matrix), and
2. pooling them into a prior that is consistent with the physical models,
   which is then constrained by the observed historical record and by
   external assessments of ECS, TCRE, forcing or ZEC.

This page records the method and the evidence behind each choice. Sections
are added as the corresponding code is merged.

## Targets

Inputs are the global-mean products of the
[scenariomip-esm-figures](https://github.com/benmsanderson/scenariomip-esm-figures)
repository (`data/`), pinned by commit in each run. For one ESM,
`cscm_calibrate.esm.targets` returns one block per time series:

| Block | Years | Variables |
|---|---|---|
| `hist` | 1850-2021 | counted once, although each scenario file repeats it |
| `<tier>` | 2022-2100 | one per ScenarioMIP tier the ESM has run |
| `abrupt-4xCO2` | 1-150 | T and N, relative to piControl |
| `1pctCO2` | 1-140 | T, relative to piControl |

Variables are GSAT (T) and net TOA flux relative to piControl (N) for
concentration-driven runs, and GSAT, CO2 concentration (C) and ocean carbon
uptake (O) for emissions-driven runs. Each block is the ensemble mean over
members.

Internal variability for each block is the standard deviation and lag-1
autocorrelation of single-member noise, divided by the square root of the
number of members. It is estimated from the linearly detrended piControl
where available (T, N), otherwise from deviations of members about the
ensemble mean (at least three members), otherwise a fixed floor.

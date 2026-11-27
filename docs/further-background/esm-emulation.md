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

Inputs are the global-mean products of the `scenariomip-esm-figures`
repository (`data/`; private for now), pinned by commit in each run. For one ESM,
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

## SCM forcing

The SCM is driven by its own forcing for each ScenarioMIP tier
(`cscm_calibrate.esm.forcing`), split into components that each ESM fit
scales separately: CO2, aerosol-radiation and aerosol-cloud interaction,
all other anthropogenic forcing, volcanic and solar. Aerosol and "other"
forcing are kept per hemisphere. Aerosol forcing is linear in its
coefficients, so the two aerosol terms come from re-running with those
coefficients set to zero.

Runs start from rest in 1850, as an ESM branching from piControl does.
Before 1850, concentrations, non-CO2 emissions and land use stay at their
1850 values, CO2 emissions and volcanic forcing are zero, and solar is
relative to its 1850-1873 mean. Forcing is relative to 1850. Starting the
SCM in 1750 instead would carry a warming and a carbon-cycle imbalance into
1850 that the ESMs do not have.

The idealised runs use the CO2 ERF of the abrupt-4xCO2 and 1pctCO2
concentrations, with all other forcing at pre-industrial.

## Emulator

`cscm_calibrate.esm.emulator` runs the CICERO-SCM thermal model (the
+epsilon configuration: upwelling-diffusion ocean with ocean heat-uptake
efficacy and no aerosol-dependent feedback) on the scaled forcing
components. An ESM is described by 13 parameters:

- eight thermal parameters (`lambda`, `rlamdo`, `akapa`, `cpi`, `W`,
  `beto`, `mixed`, `ocean_efficacy`), with uniform priors that widen the
  +epsilon prior where pilot fits reached its bounds;
- five forcing scalings (`s_co2`, `s_ari`, `s_aci`, `s_other`, `s_volc`),
  which absorb differences between the ESM's effective forcing and the
  SCM's, such as its CO2 forcing per doubling or its aerosol strength.

The emulator returns GSAT (relative to 1850-1900) and net TOA flux for each
scenario, and for abrupt-4xCO2 and 1pctCO2 driven by scaled CO2 forcing
alone. ECS and TCR are diagnosed as for the ESMs: a Gregory regression over
abrupt-4xCO2 years 1-150, and 1pctCO2 warming over years 61-80.

Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

The changes listed in this file are categorised as follows:

    - Added: new features
    - Changed: changes in existing functionality
    - Deprecated: soon-to-be removed features
    - Removed: now removed features
    - Fixed: any bug fixes
    - Security: in case of vulnerabilities.

[Unreleased]
---------------------------

Changes on this branch since the ``v0.0.1`` release.

### Added

- Post-pruning experiment workflow through ``CSCMCalibrationPipeline.run_additional_experiments_post_pruning``. Additional idealised experiments run only for members that pass pruning, with chunked execution, reuse of existing chunk outputs, and combined constraint targets for posterior weighting.
- Support for Pacific Decadal Oscillation (PDO) inputs through ``pdo_timeseries_path`` in prior configurations and ``pdo_timeseries`` in scenario setup. Monthly indices are passed to CICERO SCM as ``pdo_index_data``, with zero padding outside the available years and replacement of the ``-99.99`` missing-value code.
- Example PDO and AR7 calibration configurations, plus archived posterior samples for AR7, combined pattern/PDO, and separate PDO feedback and efficacy experiments.
- Parameter-table utilities to read prior ranges and posterior samples, calculate percentile ranges, export CSV and LaTeX tables, and plot prior/posterior histograms. A companion script generates tables and figures from calibration JSON files.
- Experiment runners for AR7-calibrated scenarios with and without natural forcings, DiamondWP emissions scenarios, flat-emissions carbon-dioxide removal and reversibility experiments, and CMIP7 ScenarioMIP branch experiments.
- Structural-ladder comparison figures for climate sensitivity, scenario projections, carbon-cycle diagnostics, rolling Gregory ECS estimates, and temperature and feedback evolution. The configurable v3 plotter supports excluding PDO results and generating missing flat-10 metrics; a companion driver compares alternative PDO feedback and efficacy configurations.
- Ladder-metric extraction to CSV, including ECS, TCR, TCRE, zero-emissions commitment, and transient peak warming.
- Diamond scenario temperature plots with quantile and median-member tables, and a utility for inspecting archived prior outputs.
- README documentation describing the post-pruning workflow and the experiment, plotting, analysis, and release-helper scripts.

### Changed

- Moved idealised experiments out of prior-ensemble generation into a separate post-pruning stage, avoiding running those experiments for members that will be discarded. The prior-ensemble and single-chunk interfaces no longer accept the former idealised-experiment arguments.
- Posterior weighting can select the combined historical and idealised target archive using ``read_idealised_constraints=True``. The lower-level weighting function also accepts an explicit ``targ_paramat_fname``.
- Shared chunk-resumption and HDF5-storage helpers are reused by prior and post-pruning experiment runners. Empty constraint-target columns are excluded from prior results.
- RCMIP protocol runs use an explicit configuration name and archived sample path, with matching output filenames. Scenario setup takes the experiment type directly, and bell-shaped scenarios extend to 2500.
- RCMIP CSV reformatting uses the current date rather than a fixed testing date.
- AR7-target evaluation and experiment-comparison scripts use updated output archives and diagnostic plotting options.
- Historical carbon-budget figures show each model configuration in its own column alongside optional Global Carbon Budget observations, with total emissions and cumulative emissions displayed consistently.

### Fixed

- Corrected the solar-forcing scenario key from ``rf_solar_data`` to ``rf_sun_data`` so supplied solar forcing is passed using the key expected by CICERO SCM.
- Updated natural CH4 and N2O input loading to read the timeseries before slicing to the simulation end year, rather than passing the unsupported ``endyear`` argument.
- Calculated steady-state natural CH4 and N2O emissions for ``esm-allGHG-piControl`` instead of prescribing zero natural emissions.
- Corrected the atmospheric carbon-flux sign in comparison plots and derived total Global Carbon Budget emissions from fossil and land-use emissions when a precomputed total is absent, accounting for cement carbonation where available.
- Aligned the year axis with the selected ``1pctCO2`` window in Seferian carbon-cycle plots and added explicit CMIP6 comparison legend entries.


[Version 0.0.1]
---------------------------

- Initial tagged release, used as the baseline for the changes listed above.
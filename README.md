<!--- --8<-- [start:description] -->
# CSCM Calibrate

Calibration pipeline for the ciceroscm simple climate model

Work in progress

## Installation

<!--- --8<-- [start:installation] -->
### As an application

If you want to use CSCM Calibrate as an application,
then we recommend using the 'locked' version of the package.
This version pins the version of all dependencies too,
which reduces the chance of installation issues
because of breaking updates to dependencies.

The locked version of CSCM Calibrate can be installed with

=== "pip"
    ```sh
    pip install 'cscm-calibrate[locked]'
    ```

### As a library

If you want to use CSCM Calibrate as a library,
for example you want to use it
as a dependency in another package/application that you're building,
then we recommend installing the package with the commands below.
This method provides the loosest pins possible of all dependencies.
This gives you, the package/application developer,
as much freedom as possible to set the versions of different packages.
However, the tradeoff with this freedom is that you may install
incompatible versions of CSCM Calibrate's dependencies
(we cannot test all combinations of dependencies,
particularly ones which haven't been released yet!).
Hence, you may run into installation issues.
If you believe these are because of a problem in CSCM Calibrate,
please [raise an issue](https://github.com/ciceroOslo/cscm-calibrate/issues).

The (non-locked) version of CSCM Calibrate can be installed with

=== "pip"
    ```sh
    pip install cscm-calibrate
    ```

Additional dependencies can be installed using

=== "pip"
    ```sh
    # To add plotting dependencies
    pip install 'cscm-calibrate[plots]'

    # To add all optional dependencies
    pip install 'cscm-calibrate[full]'
    ```

### For developers

For development, we rely on [uv](https://docs.astral.sh/uv/)
for all our dependency management.
To get started, you will need to make sure that uv is installed
([instructions here](https://docs.astral.sh/uv/getting-started/installation/)
(we found that the self-managed install was best,
particularly for upgrading uv later).

For all of our work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.
In order to create your environment, run `make virtual-environment`.

If there are any issues, the messages from the `Makefile` should guide you through.
If not, please raise an issue in the
[issue tracker](https://github.com/ciceroOslo/cscm-calibrate/issues).

For the rest of our developer docs, please see [development][development].

<!--- --8<-- [end:installation] -->

## CSCM Calibration Pipeline

This repository provides a pipeline for running, pruning, and weighting ensembles of the CICERO Simple Climate Model (SCM) for climate calibration and uncertainty quantification. The pipeline is designed to:

- Generate prior ensembles of model runs using configurable parameter distributions.
- Prune the ensemble based on observational constraints (e.g., temperature time series).
- Run any number of additional idealised or other experiments to the pruned subensemble.
- Apply posterior weighting and draw a final ensemble consistent with constraints.

The setup is also very much made to be able to "continue from where it left off", so can easily be either stopped and restart, or do a different backend setup to different constraints from a ready pruned ensemble

Configuration files and scripts decide the setup workflow.

Alongside the calibration, the engine can also be used to do runs, this has been used to run RCMIP and all experiments necessary for the paper "Pattern effects shift climate sensitivity constraints in simple climate model".

It also have several utilities to plot results from output distributions

### Main Features

- Modular pipeline: Each step (prior, prune, weight) can be run independently or as a full workflow.
- Configurable via a single JSON file (see [`tests/test-data/config_file.json`](tests/test-data/config_file.json) for an example).
- Integration with the CICERO SCM codebase.
- Includes unit and integration tests for all major components.

### Example: Running the Pipeline

```python
from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline

# Path to your config file (see tests/test-data/config_file.json for structure)
config_path = "tests/test-data/config_file.json"

pipeline = CSCMCalibrationPipeline(config_path)
pipeline._run_prior_ensemble()  # Generate prior ensemble
pipeline.prune_distribution()   # Prune ensemble based on constraints
pipeline.weight_ensemble_and_draw_write_config()  # Weight and draw final ensemble

# Or run the full pipeline
pipeline.run_full_calibration_pipeline()
```

### Configuration

The pipeline is controlled by a JSON config file. See [`tests/test-data/config_file.json`](tests/test-data/config_file.json) for a minimal working example. The config file specifies:

- Prior parameter distributions and scenario data
- Pruning constraints and variables
- Weighting and output ensemble size
- Paths to input data

### Testing

Unit and integration tests are provided in the `tests/` directory. To run all tests:

```bash
make test
```

## Scripts: runs, plots, and utilities

The tracked files in `scripts/` support project-specific workflows in addition
to the package API. Most use paths, run identifiers, calibration files, and
data locations that are set in the script itself. Review and adapt those values
before running a script. The reusable calibration workflow is exposed through
`CSCMCalibrationPipeline`; the run and plotting scripts below are primarily
research workflows rather than a stable command-line interface.

### Running experiments

- `run_full_rcmip_protocol.py` is the main RCMIP protocol runner. It also
  provides the scenario and input helpers imported by several other scripts.
- `run_ar7_scenarios.py` runs RCMIP and idealised scenario ensembles calibrated
  to AR7 targets enabled, writing natural-forcing and
  no-natural-forcing outputs.
- `run_branch_and_special_experiments.py` derives emissions-threshold branch
  experiments from a `1pctCO2` ensemble and runs the resulting scenarios.
- `run_diamond_scenarios.py` discovers DiamondWP emissions inputs and runs the
  corresponding ensemble simulations.
- `run_flat_reversibility_exps_variations.py` runs flat-emissions carbon-dioxide
  removal and reversibility experiments, and writes the input CO2 trajectories.
  It is exploratory and executes when imported.
- `run_norman_cmip7_experiments.py` runs branch-year experiments for selected
  CMIP7 ScenarioMIP pathways. It is an exploratory, import-executed script.
- `run_norman_experiments.py` runs flat-emissions experiments with linear,
  accelerating, and decelerating trajectories. It is exploratory and executes
  when imported.
- `test_pipeline_script.py` is a manual driver for the calibration pipeline:
  it creates a prior ensemble, prunes it, and draws a weighted posterior.
- `test_pipeline_script_ar7.py` is the equivalent manual driver for the
   calibration and with post-pruning experiments.

### Plotting and evaluation

- `plot_and_evaluate_w_ar7_targets.py` compares ensemble diagnostics with AR7
  WG1 Chapter 5 targets and creates plume and composite figures.
- `plot_carbon_cycle_from_dumps.py` creates carbon-cycle comparison plots from
  three output dumps, optionally overlaying observations and CMIP6 data.
- `plot_diamond_scenarios.py` groups Diamond scenario output by model, plots
  temperature ensembles, and saves quantile and median-member tables.
- `plot_flat10_vs_cumulative_emissions.py` plots flat-10 output variables
  against cumulative CO2 emissions.
- `plot_ladder_figures.py` creates the original three structural-ladder
  comparison figures. Use `plot_ladder_figures_v3.py` for new work.
- `plot_ladder_figures_v2.py` is an intermediate ladder figure generator that
  added Seferian, Gregory, and ScenarioMIP temperature and lambda plots.
- `plot_ladder_figures_v3.py` is the current configurable ladder-figure
  generator. It can exclude PDO results with `--exclude-pdo` and generates
  missing flat-10 metrics when possible. This is what generates the main figure
  of the preprint "Pattern effects shift climate sensitivity constraints in
  simple climate models"
- `plot_ladder_figures_v3_old.py` is the earlier fixed-configuration v3
  implementation, retained for reproducibility.
- `plot_ladder_figures_varying.py` configures additional PDO efficacy/lambda
  ladder experiments and delegates figure production to
  `plot_ladder_figures_v3.py`.
- `plot_sanity_checks.py` creates single-member diagnostic plots for the
  hard-coded sanity-check run identifier; it is intended for local debugging.
- `plot_various_from_experiments.py` compares pattern, no-pattern, and
  no-efficacy experiment dumps, including flat-10 analyses.

### Data preparation and analysis utilities

- `check_done_experiments_rcmip.py` reports RCMIP scenario-output files missing
  from a local output dump.
- `cleanup_comparison_figures.py` migrates comparison figures to the current
  directory and naming convention, moving, renaming, or removing duplicates.
- `extract_ladder_metrics.py` calculates ECS, TCR, TCRE, ZEC, transient peak
  warming, and related metrics for ladder experiments, writing
  `ladder_metrics_summary.csv`.
- `look_at_prior_output.py` inspects archived prior targets, parameters, and
  selected sample indices; it is an exploratory inspection tool.
- `make_latex_parameter_table_from_json.py` converts prior and posterior
  parameter JSON files into CSV and LaTex tables, with optional histogram
  panels.
- `make_rcmip_meta_files.py` generates the RCMIP model and comments metadata
  CSVs from the protocol workbook.
- `reformat_csvs_for_rcmip.py` converts raw output CSVs to RCMIP-oriented
  filenames, schema, and units.
- `adventures_in_zeb_and_chris_weightdistroland.py` is an exploratory prototype
  for iteratively deriving sample weights to match marginal distributions; it
  writes diagnostic weight plots and intentionally exits before a final result.
- `comments_ciceroscm.csv` is the RCMIP comments metadata used by the
  submission workflow.
- `meta_model_ciceroscm.csv` is the RCMIP climate-model metadata used by the
  submission workflow.

### Release and installation helpers

- `add-locked-targets-to-pyproject-toml.py` uses `uv export` to generate locked
  optional-dependency groups. It overwrites `pyproject.toml`; run it only when
  intentionally refreshing those dependency groups.
- `changelog-to-release-template.py` prints the newest Towncrier release-note
  section from `docs/changelog.md`.
- `test-install.py` imports every package submodule as an installation
  smoke test and prints the installed package version.

---

## Detailed Workflow Description

The CSCM calibration pipeline performs Bayesian calibration of the CICERO Simple Climate Model using observational constraints. The workflow consists of three main stages:

### 1. Prior Ensemble Generation

The pipeline first generates a large ensemble of model runs sampling from prior parameter distributions:

- **Parameter Sampling**: Samples parameters from uniform or custom distributions defined in the config file
- **Parallel Execution**: Runs the CICERO SCM in parallel across multiple CPU cores (configurable via `max_workers`)
- **Output Variables**: Saves time series for key climate variables (temperature, ocean heat content, radiative forcing, CO2 concentration, carbon fluxes)
- **Chunked Processing**: Processes large ensembles in chunks to manage memory (default: 10,000 runs per chunk)

**Key Config Parameters**:
- `distnums`: Total number of prior ensemble members (e.g., 6,000,000 for production runs)
- `prior_distro_dict`: Parameter ranges for sampling (climate sensitivity, ocean mixing, aerosol forcing, etc.)
- `max_workers`: Number of parallel CPU cores to use (e.g., 100 on shared HPC systems)
- `chunk_size`: Number of ensemble members per processing chunk

### 2. Ensemble Pruning

The pruning step filters the prior ensemble to remove physically implausible runs:

- **Observational Constraints**: Compares model output to historical observations (e.g., GMST time series)
- **RMSE Filtering**: Removes ensemble members that exceed a threshold RMSE relative to observations
- **Multi-Variable Support**: Can prune based on multiple observational constraints simultaneously
- **Baseline Adjustment**: Properly handles anomaly calculations relative to pre-industrial baselines

**Key Config Parameters**:
- `prune_configs`: Specifies pruning variable, observational data file, and RMSE threshold
- Example: Temperature RMSE threshold of 0.17°C over 1850-2023

### 3. Posterior Weighting and Drawing

The final stage applies importance weighting to match multiple constraints:

- **Importance Sampling**: Computes weights for each ensemble member based on fit to observational targets
- **Multi-Constraint Weighting**: Combines constraints from multiple variables (OHC, GMST, aerosol forcing, CO2, carbon fluxes)
- **Uncertainty Propagation**: Uses observational uncertainties to compute likelihood weights
- **Final Ensemble**: Draws a smaller posterior ensemble (e.g., 500 members) weighted by fit to constraints
- **Configuration Output**: Exports the drawn ensemble as a JSON configuration file for future SCM runs

**Key Config Parameters**:
- `constraint_configs`: Defines observational targets, uncertainties, and normalization periods for each variable
- `output_ensemble_size`: Number of posterior ensemble members to draw (e.g., 500)

### 4. Full Pipeline Execution

You can run all three stages sequentially:

```python
from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline

# Initialize with config and optional external constraints
pipeline = CSCMCalibrationPipeline(
    config_file="path/to/config.json",
    constraints_to_read_separately="path/to/rcmip_constraints.csv"  # Optional
)

# Run complete workflow
pipeline.run_full_calibration_pipeline()
```

Or run stages individually for debugging/development:

```python
# Stage 1: Generate prior ensemble
pipeline._run_prior_ensemble()

# Stage 2: Prune ensemble
pipeline.prune_distribution()

# Stage 3: Weight and draw posterior
pipeline.weight_ensemble_and_draw_write_config()
```

### Output Files

All output files are saved to the `output/` directory in the project root:

- **Prior ensemble**: `{variable}_{samples}_chunk_{N}_{date}_1850-2023.npy` - Time series arrays
- **Sample IDs**: `sample_ids_{samples}_chunk_{N}_{date}.npy` - Ensemble member identifiers
- **Target/Parameter matrices**: `data_{samples}_chunk_{N}_{date}.h5` - HDF5 files with constraint targets and parameter values
- **Pruned ensemble**: `valid_indices_all_chunks_{date}.npy`, `valid_sample_ids_all_chunks_{date}.npy`
- **Final ensemble**: `draw_samples_{size}_{date}.json` - Posterior ensemble configuration

### Dependencies

The pipeline requires:

- **ciceroscm**: The CICERO Simple Climate Model (must be located at `../ciceroscm` relative to project root)
- **Input Data**: Emissions, concentrations, and forcing data files (specified in config)
- **Constraint Data**: Observational time series for pruning and weighting

### Performance Considerations

- **Memory**: Large ensembles (millions of members) require chunked processing
- **CPU**: Set `max_workers` based on available cores and system sharing policies
- **Storage**: Prior ensemble files can be large (GBs per chunk); ensure adequate disk space
- **Runtime**: Full pipeline with 6M prior members can take hours to days depending on hardware

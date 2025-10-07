<!--- --8<-- [start:description] -->
# CSCM Calibrate

Calibration pipeline for the ciceroscm simple climate model

**Key info :**
[![Docs](https://readthedocs.org/projects/cscm-calibrate/badge/?version=latest)](https://cscm-calibrate.readthedocs.io)
[![Main branch: supported Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FciceroOslo%2Fcscm-calibrate%2Fmain%2Fpyproject.toml)](https://github.com/ciceroOslo/cscm-calibrate/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/pypi/l/cscm-calibrate?label=licence)](https://github.com/ciceroOslo/cscm-calibrate/blob/main/LICENCE)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/cscm-calibrate.svg)](https://pypi.org/project/cscm-calibrate/)
[![PyPI install](https://github.com/ciceroOslo/cscm-calibrate/actions/workflows/install-pypi.yaml/badge.svg?branch=main)](https://github.com/ciceroOslo/cscm-calibrate/actions/workflows/install-pypi.yaml)

**Tests :**
[![CI](https://github.com/ciceroOslo/cscm-calibrate/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/ciceroOslo/cscm-calibrate/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/ciceroOslo/cscm-calibrate/branch/main/graph/badge.svg)](https://codecov.io/gh/ciceroOslo/cscm-calibrate)

**Other info :**
[![Last Commit](https://img.shields.io/github/last-commit/ciceroOslo/cscm-calibrate.svg)](https://github.com/ciceroOslo/cscm-calibrate/commits/main)
[![Contributors](https://img.shields.io/github/contributors/ciceroOslo/cscm-calibrate.svg)](https://github.com/ciceroOslo/cscm-calibrate/graphs/contributors)
## Status

<!---

We recommend having a status line in your repo
to tell anyone who stumbles on your repository where you're up to.
Some suggested options:

- prototype: the project is just starting up and the code is all prototype
- development: the project is actively being worked on
- finished: the project has achieved what it wanted
  and is no longer being worked on, we won't reply to any issues
- dormant: the project is no longer worked on
  but we might come back to it,
  if you have questions, feel free to raise an issue
- abandoned: this project is no longer worked on
  and we won't reply to any issues
-->

- prototype: the project is just starting up and the code is all prototype

<!--- --8<-- [end:description] -->

Full documentation can be found at:
[cscm-calibrate.readthedocs.io](https://cscm-calibrate.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.

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
- Apply posterior weighting and draw a final ensemble consistent with constraints.

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

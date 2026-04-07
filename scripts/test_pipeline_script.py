"""Test script for CSCM calibration pipeline."""

import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline

# Configuration files setup
config_file = os.path.join(
    os.path.dirname(__file__), "..", "tests", "test-data", "config_file.json"
)
# Alternative config file for old model version
# config_file = os.path.join(
#     os.path.dirname(__file__), "config_file_RCMIP_run_testing.json"
# )
constraints_from_RCMIP = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "calibration_data_Sep2025",
    "rcmip_phase3_constraint_targets_with_uncertainty_v1.1.1.csv",
)

# Initialize calibration pipeline
calibration_pipeline = CSCMCalibrationPipeline(
    config_file=config_file, constraints_to_read_separately=constraints_from_RCMIP
)
print("Initialized CSCMCalibrationPipeline with config file:", config_file)
calibration_pipeline._run_prior_ensemble(
    continue_from_existing=True, #plot=True
)  # Generate prior ensemble
print(config_file)
with open(config_file) as json_config:
    configs_raw = json.load(json_config)
print(configs_raw["prior_configs"])
calibration_pipeline.prune_distribution()  # Prune ensemble based on constraints
# Weight and draw final ensemble
calibration_pipeline.weight_ensemble_and_draw_write_config()

# "nystart": 1750,
# "emstart": 1850,
# "nyend": 2023

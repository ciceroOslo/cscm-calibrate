"""Test script for CSCM calibration pipeline."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline

# Configuration files setup
config_file = os.path.join(
    os.path.dirname(__file__), "..", "tests", "test-data", "config_file.json"
)
constraints_from_RCMIP = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "calibration_data_Sep2025",
    "rcmip_phase3_constraint_targets_with_uncertainty_v1.0.0.csv",
)

# Initialize calibration pipeline
calibration_pipeline = CSCMCalibrationPipeline(
    config_file=config_file, constraints_to_read_separately=constraints_from_RCMIP
)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline

config_file = os.path.join(os.path.dirname(__file__), "..", "tests", "test-data", "config_file.json")
constraints_from_RCMIP=os.path.join("../../rcmip-phase-3/RCMIP3_input_datafiles/rcmip_phase3_constraint_targets_with_uncertainty_v1.0.0.csv")
calibration_pipeline = CSCMCalibrationPipeline(config_file=config_file, constraints_to_read_separately=constraints_from_RCMIP)

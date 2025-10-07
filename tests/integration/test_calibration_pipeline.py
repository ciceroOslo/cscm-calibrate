import os

from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline


def test_calibration_setup(test_data_dir):
    calibration_pipeline = CSCMCalibrationPipeline(os.path.join(test_data_dir, "config_file.json"))
    assert set(calibration_pipeline.configs.keys()) == set(["meta_configs", "prior_configs", "prune_configs", "constraint_configs"])
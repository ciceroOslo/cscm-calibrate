import os

from cscm_calibrate import CSCMCalibrationPipeline


def test_calibration_setup(testdir):
    calibration_pipeline = CSCMCalibrationPipeline(os.path.join(testdir, "config_file.json"))
    assert set(calibration_pipeline.cfgs.keys()) == set(["prior_configs", "prune_configs", "constraint_configs"])
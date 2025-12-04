import json
import os
import tempfile

import pytest

from cscm_calibrate.cscm_calibrate import CSCMCalibrationPipeline


class DummyConfig:
    """Minimal config for integration test."""

    @staticmethod
    def write_to(path):
        config = {
            "prior_configs": {
                "prior_distro_dict": {},
                "set_values": {},
                "input_dir": "./",
                "gases": None,
                "nat_ch4": None,
                "nat_n2o": None,
                "emis": None,
                "conc": None,
                "nystart": 1750,
                "emstart": 1850,
                "nyend": 2023,
            },
            "constraing_configs": {},
            "prune_configs": {},
            "constraint_configs": {},
            "meta_configs": {"output_ensemble_size": 1},
            "distnums": 10,
            "chunk_size": 2,
        }
        with open(path, "w") as f:
            json.dump(config, f)


def test_calibration_setup(test_data_dir):
    calibration_pipeline = CSCMCalibrationPipeline(
        os.path.join(test_data_dir, "config_file.json")
    )
    assert set(calibration_pipeline.configs.keys()) == set(
        ["meta_configs", "prior_configs", "prune_configs", "constraint_configs"]
    )

def test_pipeline_minimal(monkeypatch):
    # Create a minimal config file
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.json")
        DummyConfig.write_to(config_path)

        # Patch all external calls to avoid running heavy computation or I/O
        monkeypatch.setattr(
            "cscm_calibrate.cscm_calibrate.define_scendata_for_scm",
            lambda *a, **k: [{}],
        )
        monkeypatch.setattr(
            "cscm_calibrate.cscm_calibrate.run_prior_ensemble", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "cscm_calibrate.cscm_calibrate.prune_all_chunks", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "cscm_calibrate.cscm_calibrate.weight_ensemble_and_draw",
            lambda *a, **k: None,
        )

        pipeline = CSCMCalibrationPipeline(config_path)
        # Assert configs loaded
        assert isinstance(pipeline.configs, dict)
        assert "prior_configs" in pipeline.configs
        assert "meta_configs" in pipeline.configs
        # Assert datestr is set
        assert hasattr(pipeline, "datestr")
        # Assert methods do not raise and configs remain unchanged
        before = dict(pipeline.configs)
        pipeline._run_prior_ensemble()
        pipeline.prune_distribution()
        pipeline.weight_ensemble_and_draw_write_config()
        after = dict(pipeline.configs)
        assert before == after
        # Optionally, check that the mocked methods were called (if using mock objects)
        # e.g. monkeypatch context manager with call tracking

"""Tests for the top-level CSCMCalibrationPipeline orchestrator."""
import json
import os

import pandas as pd
import pytest

import cscm_calibrate.cscm_calibrate as cm


class TestCSCMCalibrationPipeline:
    """Unit tests for `CSCMCalibrationPipeline`.

    Uses a real test class so that the JSON config file written in setup_method
    plus the constructed pipeline instance can be reused across the tests.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        # Stub the config-distro constructor so we don't require ciceroscm.
        monkeypatch.setattr(cm, "_ConfigDistro", lambda **kw: object())

        self.tmp_path = tmp_path
        self.input_dir = tmp_path / "inputs"
        self.input_dir.mkdir()
        self.config = {
            "prior_configs": {
                # relative path -- pipeline must resolve it against config dir
                "input_dir": "inputs",
                "prior_distro_dict": {},
                "set_values": {},
                "gases": None,
                "nat_ch4": None,
                "nat_n2o": None,
                "conc": None,
                "emis": None,
                "nystart": 1750,
                "emstart": 1850,
                "nyend": 2023,
                "distnums": 10,
                "chunk_size": 5,
            },
            "constraint_configs": pd.DataFrame(
                {
                    "Variable Name": ["GMST"],
                    "Varname_short": ["GMST"],
                    "Yearstart_norm": [1850],
                    "Yearend_norm": [1900],
                    "Yearstart_change": [2000],
                    "Yearend_change": [2010],
                    "Central Value": [1.0],
                    "lower_sigma": [0.2],
                    "upper_sigma": [0.2],
                }
            ).to_dict(orient="list"),
            "prune_configs": {
                "Surface Air Ocean Blended Temperature Change": [
                    "Surface Air Ocean Blended Temperature Change",
                    "annual_averages.csv",
                    1.0,
                ]
            },
            "meta_configs": {"output_ensemble_size": 50},
        }
        self.config_path = tmp_path / "calib_config.json"
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(self.config, fh)

        # Build the pipeline once.
        self.pipeline = cm.CSCMCalibrationPipeline(str(self.config_path))

    # ----- read_in_configs -------------------------------------------------

    def test_read_in_configs_resolves_relative_input_dir(self):
        resolved = self.pipeline.configs["prior_configs"]["input_dir"]
        # The resolved path must be absolute and point at our input dir.
        assert os.path.isabs(resolved)
        assert os.path.samefile(resolved, self.input_dir)

    def test_read_in_configs_with_rcmip_csv(self, monkeypatch):
        sentinel_df = pd.DataFrame({"Variable Name": ["X"]})
        monkeypatch.setattr(
            cm, "make_constraints_config_from_RCMIP_csv", lambda **kw: sentinel_df
        )
        pipeline = cm.CSCMCalibrationPipeline(
            str(self.config_path),
            constraints_to_read_separately=str(self.tmp_path / "rcmip.csv"),
        )
        # The dataframe replaces the dict-form constraint configs.
        assert pipeline.configs["constraint_configs"] is sentinel_df

    # ----- prune_distribution ---------------------------------------------

    def test_prune_distribution_calls_prune_all_chunks(self, monkeypatch):
        captured = {}

        def fake_prune(*, total_samples, prune_lists, num_chunks, file_endstring):
            captured["total_samples"] = total_samples
            captured["prune_lists"] = prune_lists
            captured["num_chunks"] = num_chunks
            captured["file_endstring"] = file_endstring

        monkeypatch.setattr(cm, "prune_all_chunks", fake_prune)
        self.pipeline.prune_distribution()

        # distnums=10, chunk_size=5 -> 2 chunks
        assert captured["total_samples"] == 10
        assert captured["num_chunks"] == 2
        # one entry per prune_configs key, with absolute path joined onto input_dir
        assert len(captured["prune_lists"]) == 1
        entry = captured["prune_lists"][0]
        assert entry[0] == "Surface Air Ocean Blended Temperature Change"
        assert entry[1].endswith("annual_averages.csv")
        assert os.path.isabs(entry[1])
        assert entry[2] == 1.0
        assert captured["file_endstring"] == ""

    # ----- weight_ensemble_and_draw_write_config --------------------------

    def test_weight_ensemble_and_draw_write_config(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            cm,
            "weight_ensemble_and_draw",
            lambda **kw: captured.update(kw),
        )
        self.pipeline.weight_ensemble_and_draw_write_config(file_endstring="_t")
        assert captured["file_endstring"] == "_t"
        assert captured["output_ensemble_size"] == 50

    # ----- run_full_calibration_pipeline ----------------------------------

    def test_run_full_calibration_pipeline(self, monkeypatch):
        order = []
        monkeypatch.setattr(
            self.pipeline, "_run_prior_ensemble", lambda: order.append("prior")
        )
        monkeypatch.setattr(
            self.pipeline, "prune_distribution", lambda: order.append("prune")
        )
        monkeypatch.setattr(
            self.pipeline,
            "weight_ensemble_and_draw_write_config",
            lambda: order.append("weight"),
        )
        self.pipeline.run_full_calibration_pipeline()
        assert order == ["prior", "prune", "weight"]

    # ----- _run_prior_ensemble plumbing -----------------------------------

    def test_run_prior_ensemble_invokes_helpers(self, monkeypatch):
        scen = [{"scenname": "ssp245-short"}]
        monkeypatch.setattr(cm, "define_scendata_for_scm", lambda **kw: scen)

        captured = {}

        def fake_run(**kw):
            captured.update(kw)

        monkeypatch.setattr(cm, "run_prior_ensemble", fake_run)
        self.pipeline._run_prior_ensemble(continue_from_existing=True, plot=False)
        assert captured["scenariodata"] is scen
        assert captured["distnums"] == 10
        assert captured["chunk_size"] == 5
        assert captured["continue_from_existing"] is True

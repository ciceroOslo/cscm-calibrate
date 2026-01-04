from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from cscm_calibrate.run_prior_ensemble import run_prior_ensemble


def test_run_prior_ensemble_minimal(monkeypatch, tmp_path):
    # Mock testconfig with required method
    testconfig = MagicMock()
    testconfig.make_config_lists = MagicMock()

    # Create mock results DataFrame: 2 runs, 1 variable, 10 years (starting from 1743)
    variables = ["var1"]
    run_ids = [0, 1]
    years = [str(1743 + i) for i in range(10)]
    rows = []
    for run_id in run_ids:
        for var in variables:
            row = {"variable": var, "run_id": run_id}
            for y in years:
                row[y] = float(run_id * 10 + int(y))
            rows.append(row)
    mock_results = pd.DataFrame(rows)

    class DummyDistributionRun:
        def __init__(self, *a, **k):
            pass

        def run_over_distribution(self, *a, **k):
            return mock_results

        @property
        def cfgs(self):
            return [
                {
                    "pamset_udm": {"a": 1},
                    "pamset_emiconc": {"b": 2},
                    "pamset_carbon": {"c": 3},
                },
                {
                    "pamset_udm": {"a": 4},
                    "pamset_emiconc": {"b": 5},
                    "pamset_carbon": {"c": 6},
                },
            ]

    monkeypatch.setattr(
        "cscm_calibrate.run_prior_ensemble.DistributionRun", DummyDistributionRun
    )

    # Minimal calibdata and prunecfgs, matching the mock years
    # but with Yearstart/Yearend >= 1750
    calibdata = pd.DataFrame(
        {
            "Variable Name": ["var1"],
            "Yearstart_norm": [1750],
            "Yearend_norm": [1750],
            "Yearstart_change": [1750],
            "Yearend_change": [1750],
        }
    )
    prunecfgs = {"var1": ["var1"]}

    # Patch np.save and pd.HDFStore to avoid file I/O
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    class DummyStore:
        def __setitem__(self, key, value):
            if key == "targ":
                assert isinstance(value, pd.DataFrame)
            if key == "parammat":
                assert isinstance(value, pd.DataFrame)

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    # Run the function (should not raise)
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=calibdata,
        prunecfgs=prunecfgs,
        distnums=2,
        chunk_size=1,
        startdate="_test",
    )
    # Assert testconfig.make_config_lists was called
    testconfig.make_config_lists.assert_called()


def test_run_prior_ensemble_realistic_shape(monkeypatch, tmp_path):
    # Mock testconfig with required method
    testconfig = MagicMock()
    testconfig.make_config_lists = MagicMock()

    # Create mock results DataFrame: 4 runs, 2 variables, 3 years (starting from 1743)
    variables = ["var1", "var2"]
    run_ids = [0, 1, 2, 3]
    years = [str(1743 + i) for i in range(10)]
    rows = []
    for run_id in run_ids:
        for var in variables:
            row = {"variable": var, "run_id": run_id}
            for y in years:
                row[y] = float(run_id * 10 + int(y))  # unique value per run/year
            rows.append(row)
    mock_results = pd.DataFrame(rows)

    class DummyDistributionRun:
        def __init__(self, *a, **k):
            pass

        def run_over_distribution(self, *a, **k):
            return mock_results

        @property
        def cfgs(self):
            return [
                {
                    "pamset_udm": {"a": 1},
                    "pamset_emiconc": {"b": 2},
                    "pamset_carbon": {"c": 3},
                },
                {
                    "pamset_udm": {"a": 4},
                    "pamset_emiconc": {"b": 5},
                    "pamset_carbon": {"c": 6},
                },
                {
                    "pamset_udm": {"a": 7},
                    "pamset_emiconc": {"b": 8},
                    "pamset_carbon": {"c": 9},
                },
                {
                    "pamset_udm": {"a": 10},
                    "pamset_emiconc": {"b": 11},
                    "pamset_carbon": {"c": 12},
                },
            ]

    monkeypatch.setattr(
        "cscm_calibrate.run_prior_ensemble.DistributionRun", DummyDistributionRun
    )

    # Calibdata with different year logic, matching the mock years
    #  but with Yearstart/Yearend >= 1750
    calibdata = pd.DataFrame(
        {
            "Variable Name": ["var1", "var2", "var1"],
            "Yearstart_norm": [1750, 1751, 1752],
            "Yearend_norm": [1750, 1751, 1752],
            "Yearstart_change": [1750, 1751, 1752],
            "Yearend_change": [1750, 1751, 1752],
        }
    )
    prunecfgs = {"var1": ["var1"], "var2": ["var2"]}

    # Patch np.save and pd.HDFStore to avoid file I/O
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    class DummyStore:
        def __setitem__(self, key, value):
            if key == "targ":
                assert isinstance(value, pd.DataFrame)
            if key == "parammat":
                assert isinstance(value, pd.DataFrame)

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    # Run the function (should not raise)
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=calibdata,
        prunecfgs=prunecfgs,
        distnums=4,
        chunk_size=2,
        startdate="_test",
    )
    testconfig.make_config_lists.assert_called()

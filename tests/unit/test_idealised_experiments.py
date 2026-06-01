"""Tests for the idealised-experiment code paths.

Covers:
- compute_ecs_gregory (happy path, slope >= 0 unphysical branch, <10 valid years)
- make_dataframe_of_zeros
- make_constraints_config_from_RCMIP_csv 5th/50th/95th column fallback
- run_single_chunk_idealised_experiments (ECS + TCR branches)
- run_prior_ensemble idealised branch in main loop
- cscm_calibrate.CSCMCalibrationPipeline._run_prior_ensemble idealised plumbing
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import cscm_calibrate.cscm_calibrate as cm
import cscm_calibrate.run_prior_ensemble as rpe
from cscm_calibrate.run_prior_ensemble import (
    run_prior_ensemble,
    run_single_chunk_idealised_experiments,
)
from cscm_calibrate.shared_functions import (
    compute_ecs_gregory,
    make_constraints_config_from_RCMIP_csv,
    make_dataframe_of_zeros,
)


# ---------------------------------------------------------------------------
# make_dataframe_of_zeros
# ---------------------------------------------------------------------------


def test_make_dataframe_of_zeros_basic():
    df = make_dataframe_of_zeros("CH4", 1750, 1755)
    assert list(df.columns) == ["CH4"]
    assert df.index.name == "year"
    assert df.index.tolist() == [1750, 1751, 1752, 1753, 1754, 1755]
    assert (df["CH4"] == 0).all()


# ---------------------------------------------------------------------------
# make_constraints_config_from_RCMIP_csv -- 5th / 50th / 95th fallback columns
# ---------------------------------------------------------------------------


def test_make_constraints_csv_percentile_columns_fallback():
    """When Central_estimate/Lower_bound/Upper_bound are absent, the function
    falls back to the 5th / 50th / 95th percentile columns."""
    sample_df = pd.DataFrame(
        {
            "Variable": ["Heat Content|Ocean"],
            "Baseline_period": ["1971-1971"],
            "Constraint_period": ["2018-2018"],
            "5th": [200.0],
            "50th": [330.0],
            "95th": [460.0],
        }
    )
    with patch("pandas.read_csv", return_value=sample_df):
        result = make_constraints_config_from_RCMIP_csv("dummy.csv")

    assert result["Central Value"].iloc[0] == 330.0
    expected_lower = (330.0 - 200.0) / 1.6448536269514722
    expected_upper = (460.0 - 330.0) / 1.6448536269514722
    assert np.isclose(result["lower_sigma"].iloc[0], expected_lower)
    assert np.isclose(result["upper_sigma"].iloc[0], expected_upper)
    assert result["Varname_short"].iloc[0] == "OHC"


# ---------------------------------------------------------------------------
# compute_ecs_gregory
# ---------------------------------------------------------------------------


def _ecs_abrupt_df(temp_curve, ohc_curve, n_years, run_id=0):
    """Build a one-member abrupt-4xCO2 DataFrame for Gregory regression."""
    years = list(range(n_years))
    year_cols = [str(y) for y in years]
    temp_row = {"variable": "Surface Air Ocean Blended Temperature Change",
                "run_id": run_id}
    ohc_row = {"variable": "Heat Content|Ocean", "run_id": run_id}
    for y in years:
        temp_row[str(y)] = float(temp_curve(y))
        ohc_row[str(y)] = float(ohc_curve(y))
    df = pd.DataFrame([temp_row, ohc_row])
    # Need integer-string year column ordering retained
    return df, year_cols


def test_compute_ecs_gregory_happy_path():
    """With T(y) = 0.5*y and OHC(y) = 3*y - 0.5*y**2 the analytic solution
    is slope = -2, intercept = 3.5, ECS = -intercept/slope/2 = 0.875.
    """
    n = 20
    df, _ = _ecs_abrupt_df(lambda y: 0.5 * y, lambda y: 3 * y - 0.5 * y * y, n)
    ecs = compute_ecs_gregory(df, start_year=0, end_year=n - 1)
    assert ecs.name == "ECS"
    assert np.isclose(ecs.iloc[0], 0.875)


def test_compute_ecs_gregory_returns_nan_for_unphysical_positive_slope():
    """Positive Gregory slope is unphysical and must yield NaN."""
    n = 20
    # OHC accelerates -> N is increasing in dT -> positive slope.
    df, _ = _ecs_abrupt_df(lambda y: 0.5 * y, lambda y: 3 * y + 0.5 * y * y, n)
    ecs = compute_ecs_gregory(df, start_year=0, end_year=n - 1)
    assert np.isnan(ecs.iloc[0])


def test_compute_ecs_gregory_returns_nan_for_too_few_points():
    """Fewer than 10 finite (dT, N) pairs must yield NaN."""
    n = 5
    df, _ = _ecs_abrupt_df(lambda y: 0.5 * y, lambda y: 3 * y - 0.5 * y * y, n)
    ecs = compute_ecs_gregory(df, start_year=0, end_year=n - 1)
    assert np.isnan(ecs.iloc[0])


# ---------------------------------------------------------------------------
# run_single_chunk_idealised_experiments
# ---------------------------------------------------------------------------


def _make_idealised_results():
    """Build a DistributionRun.run_over_distribution-style return value with two
    scenarios: an abrupt-4xCO2 ECS scenario and a 1pctCO2 TCR scenario."""
    rows = []
    # abrupt-4xCO2: 20 years of T and OHC for ECS
    n = 20
    for run_id in (0, 1):
        temp_row = {
            "scenario": "abrupt-4xCO2",
            "variable": "Surface Air Ocean Blended Temperature Change",
            "run_id": run_id,
        }
        ohc_row = {
            "scenario": "abrupt-4xCO2",
            "variable": "Heat Content|Ocean",
            "run_id": run_id,
        }
        for y in range(n):
            temp_row[str(y)] = 0.5 * y
            ohc_row[str(y)] = 3 * y - 0.5 * y * y
        rows.extend([temp_row, ohc_row])
    # 1pctCO2 scenario: just need two specific years 1850 and 1920 for TCR
    for run_id in (0, 1):
        trow = {
            "scenario": "1pctCO2",
            "variable": "Surface Air Ocean Blended Temperature Change",
            "run_id": run_id,
        }
        trow[1850] = 0.0
        trow[1920] = 2.5 + 0.1 * run_id
        rows.append(trow)
    return pd.DataFrame(rows)


def test_run_single_chunk_idealised_experiments_ecs_and_tcr(monkeypatch, tmp_path):
    """Cover the ECS and TCR branches of `run_single_chunk_idealised_experiments`."""
    results = _make_idealised_results()

    class DummyDistributionRun:
        def __init__(self, *a, **k):
            self.cfgs = [
                {
                    "pamset_udm": {"a": 1},
                    "pamset_emiconc": {"b": 2},
                    "pamset_carbon": {"c": 3},
                }
            ]

        def run_over_distribution(self, *a, **k):
            return results

    monkeypatch.setattr(rpe, "DistributionRun", DummyDistributionRun)
    # Provide a configs JSON file that DistributionRun would normally read.
    (tmp_path / "configs_test.json").write_text("[]")

    scenariodata = [
        {
            "scenname": "abrupt-4xCO2",
            "ref_yr": 2010,
            "rf_luc_data": None,
        },
        {
            "scenname": "1pctCO2",
            # No ref_yr -> exercises the branch where ref_yr is not in keys.
            "rf_luc_data": None,
        },
    ]
    calib_data = {
        "Experiments": ["abrupt-4xCO2", "1pctCO2"],
        "Varname_short": ["ECS", "TCR"],
        "Yearstart_norm": [0, 1850],
        "Yearend_norm": [0, 1850],
        "Yearstart_change": [0, 1920],
        "Yearend_change": [19, 1920],
    }
    out = run_single_chunk_idealised_experiments(
        scenariodata=scenariodata,
        max_workers=1,
        output_dir=str(tmp_path),
        file_midstring="test",
        calib_data=calib_data,
        chunk_size=1,
    )
    assert "ECS" in out.columns
    assert "TCR" in out.columns
    # ECS analytic result (same curve as happy-path test above)
    assert np.isclose(out["ECS"].iloc[0], 0.875)
    # TCR is just T[1920] - T[1850] = 2.5 + 0.1*run_id
    assert np.allclose(out["TCR"].values, [2.5, 2.6])


# ---------------------------------------------------------------------------
# run_prior_ensemble: idealised branch in main loop (lines 228-237)
# ---------------------------------------------------------------------------


def test_run_prior_ensemble_invokes_idealised_branch(monkeypatch, tmp_path):
    """When `scenariodata_idealised_experiments` and `calibdata_idealised_experiments`
    are both provided, the idealised branch in the chunk loop must be taken and the
    targets concatenated.  Also covers the pop_keys branch in run_single_chunk
    by providing a calibdata variable that does not appear in results
    (skip_idealised_experiments=True will then strip empty entries)."""
    monkeypatch.setattr(rpe, "get_project_root", lambda: str(tmp_path))

    # Output dir set up; configs file not needed because we patch
    # _generate_prior_ensemble_parameters.
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(
        rpe, "_generate_prior_ensemble_parameters", lambda **kw: None
    )

    # Construct a minimal results DataFrame: 2 runs, var1 only
    years = [str(1743 + i) for i in range(10)]
    rows = []
    for rid in (0, 1):
        row = {"variable": "var1", "run_id": rid}
        for y in years:
            row[y] = float(rid * 10 + int(y))
        rows.append(row)
    mock_results = pd.DataFrame(rows)

    cfgs = [
        {
            "pamset_udm": {"a": k},
            "pamset_emiconc": {"b": k + 1},
            "pamset_carbon": {"c": k + 2},
        }
        for k in range(2)
    ]

    class DummyDistributionRun:
        def __init__(self, *a, **k):
            self.cfgs = cfgs

        def run_over_distribution(self, *a, **k):
            return mock_results

    monkeypatch.setattr(rpe, "DistributionRun", DummyDistributionRun)
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    class DummyStore:
        def __setitem__(self, key, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    # Patch out the inner idealised-experiments runner so we don't have to set up
    # ECS/TCR data again; just record that it got called and return a small frame.
    captured_idealised = {}

    def fake_idealised(scenariodata, max_workers, output_dir, file_midstring,
                       calib_data, chunk_size=10000):
        captured_idealised["called"] = True
        captured_idealised["scenarios"] = scenariodata
        # Return a frame indexed [0, 1] with a single ECS column
        return pd.DataFrame({"ECS": [1.0, 2.0]}, index=[0, 1])

    monkeypatch.setattr(rpe, "run_single_chunk_idealised_experiments", fake_idealised)

    # calibdata with an extra variable ("missing_var") triggers the pop_keys
    # branch in run_single_chunk because mock_results has no rows for it.
    calibdata = pd.DataFrame(
        {
            "Variable Name": ["var1", "missing_var"],
            "Yearstart_norm": [1750, 1750],
            "Yearend_norm": [1750, 1750],
            "Yearstart_change": [1750, 1750],
            "Yearend_change": [1750, 1750],
        }
    )

    testconfig = MagicMock()
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=calibdata,
        prunecfgs={"var1": ["var1"]},
        distnums=2,
        chunk_size=2,
        scenariodata_idealised_experiments=[{"scenname": "abrupt-4xCO2"}],
        calibdata_idealised_experiments={"Experiments": ["abrupt-4xCO2"]},
    )
    assert captured_idealised.get("called") is True


# ---------------------------------------------------------------------------
# cscm_calibrate.CSCMCalibrationPipeline._run_prior_ensemble idealised plumbing
# ---------------------------------------------------------------------------


def test_cscm_calibrate_run_prior_ensemble_idealised_plumbing(
    monkeypatch, tmp_path,
):
    """Drive `_run_prior_ensemble` with `constraint_configs_idealised` to cover
    both the `esm`-prefixed and the concentration-driven branches that build
    the per-experiment scenario data."""
    monkeypatch.setattr(cm, "_ConfigDistro", lambda **kw: object())

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    config = {
        "prior_configs": {
            "input_dir": str(input_dir),
            "prior_distro_dict": {},
            "set_values": {},
            "gases": None,
            "nat_ch4": None,
            "nat_n2o": None,
            "conc": "historical_conc.txt",
            "emis": "historical_em.txt",
            "nystart": 1750,
            "emstart": 1850,
            "nyend": 2023,
            "distnums": 4,
            "chunk_size": 2,
        },
        "constraint_configs": pd.DataFrame(
            {"Variable Name": ["GMST"]}
        ).to_dict(orient="list"),
        "constraint_configs_idealised": {
            "Experiments": ["abrupt-4xCO2", "esm-1pctCO2"],
            "Varname_short": ["ECS", "TCRE"],
            "Yearstart_norm": [1850, 1850],
            "Yearend_norm": [1850, 1850],
            "Yearstart_change": [1850, 1920],
            "Yearend_change": [2020, 1920],
        },
        "prune_configs": {},
        "meta_configs": {"output_ensemble_size": 50},
    }
    cfg_path = tmp_path / "calib_config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh)

    # `define_scendata_for_scm` is called once for the historical scenario, then
    # once per idealised experiment. Return a unique sentinel per call so we can
    # verify what gets passed downstream.
    call_log = []

    def fake_define(**kw):
        call_log.append(kw)
        return [
            {
                "scenname": "sentinel",
                "rf_solar_data": None,
                "rf_volc_data": None,
                "rf_luc_data": None,
            }
        ]

    monkeypatch.setattr(cm, "define_scendata_for_scm", fake_define)

    captured = {}

    def fake_run(**kw):
        captured.update(kw)

    monkeypatch.setattr(cm, "run_prior_ensemble", fake_run)

    pipeline = cm.CSCMCalibrationPipeline(str(cfg_path))
    pipeline._run_prior_ensemble(continue_from_existing=False, plot=False)

    # 1 historical + 2 idealised = 3 calls
    assert len(call_log) == 3
    # Idealised scenario list has two entries with scenname tagged.
    idealised = captured["scenariodata_idealised_experiments"]
    assert isinstance(idealised, list)
    assert len(idealised) == 2
    scennames = [s["scenname"] for s in idealised]
    assert scennames == ["abrupt-4xCO2", "esm-1pctCO2"]
    # The esm-prefixed experiment must be emission-driven (conc_run False), the
    # other concentration-driven (conc_run True).
    by_name = {s["scenname"]: s for s in idealised}
    assert by_name["esm-1pctCO2"]["conc_run"] is False
    assert by_name["abrupt-4xCO2"]["conc_run"] is True
    # ref_yr is min(nyend-1, 2010): 2010 for abrupt-4xCO2, 1919 for esm-1pctCO2
    assert by_name["abrupt-4xCO2"]["ref_yr"] == 2010
    assert by_name["esm-1pctCO2"]["ref_yr"] == 1919
    # calibdata_idealised passed through
    assert (
        captured["calibdata_idealised_experiments"]
        is pipeline.configs["constraint_configs_idealised"]
    )


def test_cscm_calibrate_run_prior_ensemble_no_idealised(monkeypatch, tmp_path):
    """When `constraint_configs_idealised` is absent, both idealised kwargs
    forwarded to `run_prior_ensemble` must be None."""
    monkeypatch.setattr(cm, "_ConfigDistro", lambda **kw: object())
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    config = {
        "prior_configs": {
            "input_dir": str(input_dir),
            "prior_distro_dict": {},
            "set_values": {},
            "gases": None,
            "nat_ch4": None,
            "nat_n2o": None,
            "conc": "historical_conc.txt",
            "emis": "historical_em.txt",
            "nystart": 1750,
            "emstart": 1850,
            "nyend": 2023,
            "distnums": 4,
            "chunk_size": 2,
        },
        "constraint_configs": pd.DataFrame(
            {"Variable Name": ["GMST"]}
        ).to_dict(orient="list"),
        "prune_configs": {},
        "meta_configs": {"output_ensemble_size": 50},
    }
    cfg_path = tmp_path / "calib_config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh)

    monkeypatch.setattr(cm, "define_scendata_for_scm", lambda **kw: [{}])
    captured = {}
    monkeypatch.setattr(cm, "run_prior_ensemble", lambda **kw: captured.update(kw))

    pipeline = cm.CSCMCalibrationPipeline(str(cfg_path))
    pipeline._run_prior_ensemble()
    assert captured["scenariodata_idealised_experiments"] is None
    assert captured["calibdata_idealised_experiments"] is None

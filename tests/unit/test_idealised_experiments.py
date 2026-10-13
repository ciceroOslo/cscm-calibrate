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
)
from cscm_calibrate.run_extra_experiments_post_pruning import (
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
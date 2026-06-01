import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd

from cscm_calibrate.shared_functions import (
    make_config_distro_json,
    make_constraints_config_from_RCMIP_csv,
    rmse,
)


def test_rmse_basic_calculation():
    """Test RMSE calculation with simple arrays."""
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])
    result = rmse(obs, mod)
    expected = np.sqrt(np.sum((obs - mod) ** 2) / len(obs))
    print(result)
    print(expected)
    assert np.isclose(result, expected)


def test_rmse_perfect_match():
    """Test RMSE when observed and modeled are identical."""
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.0, 2.0, 3.0])
    result = rmse(obs, mod)
    assert result == 0.0


def test_rmse_single_value():
    """Test RMSE with single values."""
    obs = np.array([5.0])
    mod = np.array([7.0])
    result = rmse(obs, mod)
    assert result == 2.0


def test_make_config_distro_json_basic():
    """Test basic functionality of make_config_distro_json without fixtures."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [3, 26]])
    print(matrix.shape)
    parameter_names = ["param1", "lambda", "beta_f", "rs_tim1"]
    json_name = "test_config.json"

    make_config_distro_json(matrix, parameter_names, json_name, output_dir="data")

    assert os.path.exists("data/test_config.json")
    with open("data/test_config.json", encoding="utf-8") as rfile:
        config_list = json.load(rfile)
    assert len(config_list) == 2  # Two configurations
    print(config_list)
    assert "pamset_udm" in config_list[0]
    assert "pamset_emiconc" in config_list[0]
    assert "pamset_carbon" in config_list[0]
    assert "Index" in config_list[0]
    os.remove("data/test_config.json")
    assert config_list[0]["pamset_udm"]["lambda"] == 3.0
    assert config_list[1]["pamset_carbon"]["beta_f"] == 6.0
    assert config_list[1]["pamset_emiconc"]["param1"] == 2.0
    assert config_list[1]["pamset_carbon"]["rs_tim1"] == 26.0


def test_make_constraints_config_from_RCMIP_csv():
    """Test parsing RCMIP CSV constraints into config DataFrame."""
    # Sample CSV data as DataFrame
    sample_df = pd.DataFrame(
        {
            "Variable": ["Atmospheric Concentrations|CO2"],
            "Baseline_period": ["1850-1900"],
            "Constraint_period": ["2000-2010"],
            "Central_estimate": [400.0],
            "Lower_bound": [380.0],
            "Upper_bound": [420.0],
        }
    )

    with patch("pandas.read_csv", return_value=sample_df):
        result = make_constraints_config_from_RCMIP_csv("dummy_path.csv")

    # Check the resulting DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["Variable Name"].iloc[0] == "Atmospheric Concentrations|CO2"
    assert result["Varname_short"].iloc[0] == "CO2conc"
    assert result["Yearstart_norm"].iloc[0] == 1850
    assert result["Yearend_norm"].iloc[0] == 1900
    assert result["Yearstart_change"].iloc[0] == 2000
    assert result["Yearend_change"].iloc[0] == 2010
    assert result["Central Value"].iloc[0] == 400.0
    # Check sigma calculations (using SIGMA_TO_90PERCENT ≈ 1.645)
    expected_lower_sigma = (400.0 - 380.0) / 1.6448536269514722
    expected_upper_sigma = (420.0 - 400.0) / 1.6448536269514722
    assert np.isclose(result["lower_sigma"].iloc[0], expected_lower_sigma)
    assert np.isclose(result["upper_sigma"].iloc[0], expected_upper_sigma)
    assert result["run_experiments"].iloc[0] == "historical"


def test_make_constraints_csv_uses_rcmip_name_mapping():
    """Variable names listed in RCMIP_NAME_MAPPING are translated and the short
    name is looked up under the translated name."""
    sample_df = pd.DataFrame(
        {
            "Variable": ["Effective Radiative Forcing|Aerosols"],
            "Baseline_period": ["1750-1750"],
            "Constraint_period": ["2014-2014"],
            "Central_estimate": [-1.3],
            "Lower_bound": [-2.0],
            "Upper_bound": [-0.6],
        }
    )
    with patch("pandas.read_csv", return_value=sample_df):
        result = make_constraints_config_from_RCMIP_csv("dummy.csv")

    assert (
        result["Variable Name"].iloc[0]
        == "Effective Radiative Forcing|Anthropogenic|Aerosol"
    )
    assert result["Varname_short"].iloc[0] == "ERFaer"


def test_make_constraints_csv_invalid_baseline_period_falls_back():
    """A non-numeric Baseline_period triggers the ValueError fallback to 1750/1750."""
    sample_df = pd.DataFrame(
        {
            "Variable": ["Atmospheric Concentrations|CO2"],
            "Baseline_period": ["not-a-period"],
            "Constraint_period": ["2000-2010"],
            "Central_estimate": [400.0],
            "Lower_bound": [380.0],
            "Upper_bound": [420.0],
        }
    )
    with patch("pandas.read_csv", return_value=sample_df):
        result = make_constraints_config_from_RCMIP_csv("dummy.csv")

    assert result["Yearstart_norm"].iloc[0] == 1750
    assert result["Yearend_norm"].iloc[0] == 1750


def test_make_config_distro_json_uses_index_list(tmp_path):
    """Explicit index_list values must end up as the configurations' Index field."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    parameter_names = ["lambda", "param1"]
    json_name = "with_index.json"
    make_config_distro_json(
        matrix,
        parameter_names,
        json_name,
        index_list=["alpha", "beta"],
        output_dir=str(tmp_path),
    )
    with open(tmp_path / json_name, encoding="utf-8") as fh:
        config_list = json.load(fh)
    assert [c["Index"] for c in config_list] == ["alpha", "beta"]

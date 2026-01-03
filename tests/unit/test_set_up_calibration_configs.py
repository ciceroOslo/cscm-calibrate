import os
from unittest.mock import patch

import pandas as pd
import pytest

from cscm_calibrate.set_up_calibration_configs_and_run import (
    define_scendata_for_scm,
    get_df_from_input_w_data_handler,
)


def test_invalid_case_type_raises_value_error():
    """Test that invalid case_type raises ValueError."""
    with pytest.raises(ValueError, match="case_type: invalid must be one of"):
        get_df_from_input_w_data_handler(
            input_concrete="file.txt",
            test_data_dir="/test/dir",
            expected_string="expected.txt",
            case_type="invalid",
        )


@patch("cscm_calibrate.set_up_calibration_configs_and_run.input_handler")
def test_non_dataframe_result_raises_type_error(mock_input_handler):
    """Test that non-DataFrame result from input_handler raises TypeError."""
    mock_input_handler.read_natural_emissions.return_value = "not a dataframe"
    with pytest.raises(TypeError, match="must be either a str, a path or a DataFrame"):
        get_df_from_input_w_data_handler(
            input_concrete="file.txt",
            test_data_dir="/test/dir",
            expected_string="expected.txt",
            case_type="CH4",
        )


def test_dataframe_input_returns_as_is():
    """Test that DataFrame input is returned unchanged."""
    df = pd.DataFrame({"test": [1, 2, 3]})
    result = get_df_from_input_w_data_handler(
        input_concrete=df,
        test_data_dir="/test/dir",
        expected_string="expected.txt",
        case_type="CH4",
    )
    pd.testing.assert_frame_equal(result, df)


def test_valid_case_type_with_mock_dataframe():
    """Test that valid case_type with mocked DataFrame works."""
    mock_df = pd.DataFrame({"year": [2020], "value": [1.0]})

    result = get_df_from_input_w_data_handler(
        input_concrete=mock_df,
        test_data_dir="/test/dir",
        expected_string="default.txt",
        case_type="CH4",
    )

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, mock_df)

    result = get_df_from_input_w_data_handler(
        input_concrete=None,
        test_data_dir="/test/dir",
        expected_string=mock_df,
        case_type="CH4",
    )

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, mock_df)


def test_define_scendata_for_scm():
    """Test that define_scendata_for_scm returns a DataFrame with expected columns."""
    scen_data = define_scendata_for_scm(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "calibration_data_RCMIP"
        ),
        sunvolc=1,
        rf_volc_file="VOLC_RCMIP_historical_RCMIP3.txt",
        rf_solar_file="solar_RCMIP_historical_RCMIP3.txt",
        rf_luc_file="LUCalbedo_RCMIP_IGCC.txt",
    )

    assert isinstance(scen_data, list)
    assert isinstance(scen_data[0], dict)
    expected_keys = set(
        [
            "gaspam_data",
            "emstart",
            "conc_run",
            "nystart",
            "nyend",
            "concentrations_data",
            "emissions_data",
            "nat_ch4_data",
            "nat_n2o_data",
            "idtm",
            "scenname",
            "sunvolc",
            "rf_solar",
            "rf_volc",
            "rf_luc",
        ]
    )
    assert expected_keys == set(scen_data[0].keys())

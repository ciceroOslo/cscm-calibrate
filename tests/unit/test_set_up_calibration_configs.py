from unittest.mock import patch

import pandas as pd
import pytest

from cscm_calibrate.set_up_calibration_configs_and_run import (
    get_df_from_input_w_data_handler,
)


class TestGetDfFromInputWDataHandler:
    """Unit tests for get_df_from_input_w_data_handler function, focusing on error handling."""

    def test_invalid_case_type_raises_value_error(self):
        """Test that invalid case_type raises ValueError."""
        with pytest.raises(ValueError, match="case_type: invalid must be one of"):
            get_df_from_input_w_data_handler(
                input_concrete="file.txt",
                test_data_dir="/test/dir",
                expected_string="expected.txt",
                case_type="invalid",
            )

    @patch("cscm_calibrate.set_up_calibration_configs_and_run.input_handler")
    def test_non_dataframe_result_raises_type_error(self, mock_input_handler):
        """Test that non-DataFrame result from input_handler raises TypeError."""
        mock_input_handler.read_natural_emissions.return_value = "not a dataframe"

        with pytest.raises(
            TypeError, match="must be either a str, a path or a DataFrame"
        ):
            get_df_from_input_w_data_handler(
                input_concrete="file.txt",
                test_data_dir="/test/dir",
                expected_string="expected.txt",
                case_type="CH4",
            )

    def test_dataframe_input_returns_as_is(self):
        """Test that DataFrame input is returned unchanged."""
        df = pd.DataFrame({"test": [1, 2, 3]})
        result = get_df_from_input_w_data_handler(
            input_concrete=df,
            test_data_dir="/test/dir",
            expected_string="expected.txt",
            case_type="CH4",
        )
        pd.testing.assert_frame_equal(result, df)

    @patch("cscm_calibrate.set_up_calibration_configs_and_run.input_handler")
    def test_valid_case_type_with_mock_dataframe(self, mock_input_handler):
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

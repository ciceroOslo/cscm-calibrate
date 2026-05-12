import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import cscm_calibrate.set_up_calibration_configs_and_run as setup_mod
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
    # The conc branch returns the input_handler value directly without slicing,
    # so it cleanly hits the TypeError raise.
    mock_input_handler.read_inputfile.return_value = "not a dataframe"
    with pytest.raises(TypeError, match="must be either a str, a path or a DataFrame"):
        get_df_from_input_w_data_handler(
            input_concrete="file.txt",
            test_data_dir="/test/dir",
            expected_string="expected.txt",
            case_type="conc",
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


def test_define_scendata_for_scm(monkeypatch, tmp_path):
    """Mocked happy path: each helper returns a small DataFrame, sunvolc=1
    reads two-column whitespace files from disk."""
    # mock the helper called by define_scendata_for_scm so we don't need
    # real RCMIP files
    stub_df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(
        setup_mod,
        "get_df_from_input_w_data_handler",
        lambda input_concrete, test_data_dir, expected_string, **kw: stub_df,
    )

    # write the three sunvolc forcing files as whitespace-separated
    for fname in ("rfvolc.txt", "rfsolar.txt", "rfluc.txt"):
        (tmp_path / fname).write_text("1750 0.0\n1751 0.1\n1752 -0.1\n")

    scen_data = define_scendata_for_scm(
        str(tmp_path),
        sunvolc=1,
        rf_volc_file="rfvolc.txt",
        rf_solar_file="rfsolar.txt",
        rf_luc_file="rfluc.txt",
    )

    assert isinstance(scen_data, list) and len(scen_data) == 1
    entry = scen_data[0]
    expected_keys = {
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
    }
    assert set(entry.keys()) == expected_keys
    assert entry["sunvolc"] == 1
    # forcing files were loaded as DataFrames
    for k in ("rf_solar", "rf_volc", "rf_luc"):
        assert isinstance(entry[k], pd.DataFrame)
        assert entry[k].shape == (3, 2)


def test_define_scendata_for_scm_sunvolc_zero(monkeypatch, tmp_path):
    """sunvolc=0 must skip all forcing-file reads."""
    stub_df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(
        setup_mod,
        "get_df_from_input_w_data_handler",
        lambda input_concrete, test_data_dir, expected_string, **kw: stub_df,
    )

    def _fail_read(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("pd.read_csv must not be called when sunvolc=0")

    monkeypatch.setattr(pd, "read_csv", _fail_read)

    scen_data = define_scendata_for_scm(str(tmp_path), sunvolc=0)
    entry = scen_data[0]
    assert entry["rf_solar"] is None
    assert entry["rf_volc"] is None
    assert entry["rf_luc"] is None


def test_define_scendata_for_scm_sunvolc_one_missing_files(monkeypatch, tmp_path):
    """sunvolc=1 with non-existent filenames keeps rf_*_data as None."""
    stub_df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(
        setup_mod,
        "get_df_from_input_w_data_handler",
        lambda input_concrete, test_data_dir, expected_string, **kw: stub_df,
    )
    scen_data = define_scendata_for_scm(
        str(tmp_path),
        sunvolc=1,
        rf_volc_file="absent_volc.txt",
        rf_solar_file="absent_solar.txt",
        rf_luc_file="absent_luc.txt",
    )
    entry = scen_data[0]
    assert entry["rf_solar"] is None
    assert entry["rf_volc"] is None
    assert entry["rf_luc"] is None


def test_case_type_n2o_uses_loc_slice(monkeypatch):
    """N2O/CH4 branch slices by .loc[:nyend] after read_natural_emissions."""
    df = pd.DataFrame({"value": [10.0, 11.0, 12.0, 13.0]}, index=[2020, 2021, 2022, 2023])
    mock_ih = MagicMock()
    mock_ih.read_natural_emissions.return_value = df
    monkeypatch.setattr(setup_mod, "input_handler", mock_ih)

    result = get_df_from_input_w_data_handler(
        input_concrete="natural.txt",
        test_data_dir="/data",
        expected_string="default.txt",
        nyend=2022,
        case_type="N2O",
    )
    mock_ih.read_natural_emissions.assert_called_once()
    # call args: first positional should be joined path
    called_path = mock_ih.read_natural_emissions.call_args.args[0]
    assert called_path == os.path.join("/data", "natural.txt")
    # .loc[:2022] keeps three rows
    assert list(result.index) == [2020, 2021, 2022]


def test_case_type_emis_renames_co2_columns(monkeypatch):
    """emis branch renames CO2 -> CO2_FF and CO2.1 -> CO2_AFOLU."""
    df = pd.DataFrame({"CO2": [1.0], "CO2.1": [2.0], "Other": [3.0]})

    class DummyHandler:
        def __init__(self, *a, **k):
            self.init_args = (a, k)

        def read_emissions(self, path):
            self.path = path
            return df.copy()

    mock_ih = MagicMock()
    mock_ih.InputHandler = DummyHandler
    monkeypatch.setattr(setup_mod, "input_handler", mock_ih)

    result = get_df_from_input_w_data_handler(
        input_concrete="em.txt",
        test_data_dir="/data",
        expected_string="default.txt",
        case_type="emis",
    )
    assert "CO2_FF" in result.columns
    assert "CO2_AFOLU" in result.columns
    assert "CO2" not in result.columns
    assert "CO2.1" not in result.columns


def test_case_type_conc_calls_read_inputfile_with_cut_years_false(monkeypatch):
    """conc branch must call read_inputfile with cut_years=False."""
    df = pd.DataFrame({"x": [1, 2]})
    mock_ih = MagicMock()
    mock_ih.read_inputfile.return_value = df
    monkeypatch.setattr(setup_mod, "input_handler", mock_ih)

    result = get_df_from_input_w_data_handler(
        input_concrete="conc.txt",
        test_data_dir="/data",
        expected_string="default.txt",
        case_type="conc",
    )
    mock_ih.read_inputfile.assert_called_once()
    assert mock_ih.read_inputfile.call_args.kwargs == {"cut_years": False}
    pd.testing.assert_frame_equal(result, df)


def test_case_type_gaspam_calls_read_components(monkeypatch):
    """gaspam branch routes through read_components."""
    df = pd.DataFrame({"gas": ["CO2"], "lifetime": [120.0]})
    mock_ih = MagicMock()
    mock_ih.read_components.return_value = df
    monkeypatch.setattr(setup_mod, "input_handler", mock_ih)

    result = get_df_from_input_w_data_handler(
        input_concrete="gaspam.txt",
        test_data_dir="/data",
        expected_string="default.txt",
        case_type="gaspam",
    )
    mock_ih.read_components.assert_called_once()
    pd.testing.assert_frame_equal(result, df)


def test_case_type_gases_falls_through_to_read_components(monkeypatch):
    """The catch-all 'gases' branch also uses read_components."""
    df = pd.DataFrame({"gas": ["N2O"]})
    mock_ih = MagicMock()
    mock_ih.read_components.return_value = df
    monkeypatch.setattr(setup_mod, "input_handler", mock_ih)

    result = get_df_from_input_w_data_handler(
        input_concrete="gases.txt",
        test_data_dir="/data",
        expected_string="default.txt",
        case_type="gases",
    )
    mock_ih.read_components.assert_called_once()
    pd.testing.assert_frame_equal(result, df)


"""
Set up calibration configs for running calibrations
"""

import os
import sys

import pandas as pd

# Get path to ciceroscm - one level up from project root
cscm_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ciceroscm")
)

sys.path.insert(0, os.path.join(cscm_path, "src"))


from ciceroscm import input_handler  # noqa: E402


def get_df_from_input_w_data_handler(  # noqa: PLR0913
    input_concrete,
    test_data_dir,
    expected_string,
    nyend=2023,
    nystart=1750,
    emstart=1850,
    case_type="CH4",
):
    """
    Load and process input data for calibration

    Loads and processes input data for calibration
    using the input_handler module.

    Parameters
    ----------
    input_concrete : str, os.PathLike, pd.DataFrame or None
        The input data, which can be a file path, DataFrame, or None.
        If None, uses `expected_string`.
    test_data_dir : str or os.PathLike
        Directory where test data files are located.
    expected_string : str
        Default filename or identifier to use if `input_concrete` is None.
    nyend : int, optional
        End year for the data (default is 2023).
    nystart : int, optional
        Start year for the data (default is 1750).
    emstart : int, optional
        Emissions start year (default is 1850).
    case_type : str, optional
        Type of data to process. Must be one of
        ["CH4", "N2O", "emis", "conc", "gases"] (default is "CH4").

    Returns
    -------
    pd.DataFrame
        Processed input data as a pandas DataFrame.

    Raises
    ------
    ValueError
        If `case_type` is not a valid option.
    TypeError
        If the processed input is not a pandas DataFrame.
    """
    valid = ["CH4", "N2O", "emis", "conc", "gases", "gaspam"]
    if case_type not in valid:
        raise ValueError(  # noqa: TRY003
            f"case_type: {case_type} must be one of the valid choices {valid}"
        )
    if input_concrete is None:
        input_concrete = expected_string
    if isinstance(input_concrete, str):
        input_concrete = os.path.join(test_data_dir, input_concrete)
    if isinstance(input_concrete, (str, os.PathLike)):
        if case_type in ["CH4", "N2O"]:
            input_concrete = input_handler.read_natural_emissions(
                input_concrete, case_type, endyear=nyend
            )
        elif case_type == "emis":
            ih = input_handler.InputHandler(
                {"nyend": nyend, "nystart": nystart, "emstart": emstart}
            )
            input_concrete = ih.read_emissions(input_concrete)
            input_concrete.rename(
                columns={"CO2": "CO2_FF", "CO2.1": "CO2_AFOLU"}, inplace=True
            )
        elif case_type == "conc":
            # RCMIP format concentration file with 4 header rows
            input_concrete = input_handler.read_inputfile(
                input_concrete, cut_years=False
            )
        elif case_type == "gaspam":
            # Gas parameters file with 1 header row
            input_concrete = input_handler.read_components(input_concrete)
        else:
            input_concrete = input_handler.read_components(input_concrete)
    if not isinstance(input_concrete, pd.DataFrame):
        raise TypeError(  # noqa: TRY003
            f"input_concrete for {case_type} must be either a str, a path or a DataFrame"  # noqa: E501
        )
    return input_concrete


def define_scendata_for_scm(  # noqa: PLR0913
    test_data_dir,
    gaspam=None,
    df_nat_ch4=None,
    df_nat_n2o=None,
    df_conc=None,
    df_emis=None,
    nyend=2023,
    nystart=1750,
    emstart=1850,
    sunvolc=0,
    rf_volc_file=None,
    rf_solar_file=None,
    rf_luc_file=None,
):
    """
    Prepare and returns scenario data for SCM

    Prepare and returns scenario data for SCM
    (Simple Climate Model ciceroscm) calibration runs.

    This function loads and processes various
    input datasets required for SCM calibration,
    including gas parameters, natural emissions,
    concentrations, and historical emissions.
    It returns a list containing a dictionary with
    all relevant scenario data.

    Parameters
    ----------
    test_data_dir : str
        Path to the directory containing test data files.
    gaspam : pandas.DataFrame or None, optional
        DataFrame containing gas parameter data.
        If None, data will be loaded from file.
    df_nat_ch4 : pandas.DataFrame or None, optional
        DataFrame containing natural CH4 emissions data.
        If None, data will be loaded from file.
    df_nat_n2o : pandas.DataFrame or None, optional
        DataFrame containing natural N2O emissions data.
        If None, data will be loaded from file.
    df_conc : pandas.DataFrame or None, optional
        DataFrame containing historical gas concentrations.
        If None, data will be loaded from file.
    df_emis : pandas.DataFrame or None, optional
        DataFrame containing historical emissions data.
        If None, data will be loaded from file.
    nyend : int, optional
        End year for the scenario data (default is 2023).
    nystart : int, optional
        Start year for the scenario data (default is 1750).
    emstart : int, optional
        Start year for emissions data (default is 1850).
    sunvolc : int, optional
        Flag for including solar and volcanic forcing (default is 0).
    rf_volc_file : str or None, optional
        Filename for volcanic radiative forcing data.
    rf_solar_file : str or None, optional
        Filename for solar radiative forcing data.
    rf_luc_file : str or None, optional
        Filename for land use change albedo forcing data.

    Returns
    -------
    scenariodata : list of dict
        A list containing a single dictionary with all
        scenario data required for SCM calibration.
        The dictionary includes gas parameters,
        natural emissions, concentrations, emissions,
        and scenario metadata.
    """
    gaspam = get_df_from_input_w_data_handler(
        gaspam,
        test_data_dir,
        "gases_vupdate_2024_WMO_added_new.txt",
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="gaspam",
    )

    # CH4-block
    df_nat_ch4 = get_df_from_input_w_data_handler(
        df_nat_ch4,
        test_data_dir,
        "natemis_CH4_ode_method_from_Sep2025_updates.txt",
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="CH4",
    )
    df_nat_n2o = get_df_from_input_w_data_handler(
        df_nat_n2o,
        test_data_dir,
        "natemis_N2O_ode_method_from_Sep2025_updates.txt",
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="N2O",
    )
    df_conc = get_df_from_input_w_data_handler(
        df_conc,
        test_data_dir,
        "historical_conc_gases_vupdate_2024_WMO_added_new.txt",
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="conc",
    )
    df_emis = get_df_from_input_w_data_handler(
        df_emis,
        test_data_dir,
        "historical_em_gases_vupdate_2024_WMO_added_new.txt",
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="emis",
    )

    # Load solar and volcanic forcing data if sunvolc flag is set
    rf_solar_data = None
    rf_volc_data = None
    rf_luc_data = None

    if sunvolc == 1:
        if rf_solar_file:
            solar_path = os.path.join(test_data_dir, rf_solar_file)
            if os.path.exists(solar_path):
                rf_solar_data = pd.read_csv(solar_path, header=None, sep=r"\s+")

        if rf_volc_file:
            volc_path = os.path.join(test_data_dir, rf_volc_file)
            if os.path.exists(volc_path):
                rf_volc_data = pd.read_csv(volc_path, header=None, sep=r"\s+")

        if rf_luc_file:
            luc_path = os.path.join(test_data_dir, rf_luc_file)
            if os.path.exists(luc_path):
                rf_luc_data = pd.read_csv(luc_path, header=None, sep=r"\s+")

    scenariodata = [
        {
            "gaspam_data": gaspam,
            "emstart": 1850,
            "conc_run": False,
            "nystart": 1750,
            "nyend": nyend,
            "concentrations_data": df_conc,
            "emissions_data": df_emis,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm": 24,
            "scenname": "ssp245-short",
            "sunvolc": sunvolc,
            "rf_solar": rf_solar_data,
            "rf_volc": rf_volc_data,
            "rf_luc": rf_luc_data,
        }
    ]
    return scenariodata

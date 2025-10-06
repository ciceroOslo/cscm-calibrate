import sys
import re
import os
import numpy as np
import pandas as pd
import pandas.testing as pdt
import warnings

from ciceroscm import CICEROSCM

from ciceroscm import input_handler

cscm_path = os.path.join("..", "..", "..", "ciceroscm")

sys.path.insert(0,os.path.join(cscm_path, 'src'))

def get_df_from_input_w_data_handler(input_concrete, test_data_dir, expected_string, nyend=2023, nystart=1750, emstart= 1850, case_type = "CH4"):
    valid = ["CH4", "N2O", "emis", "conc", "gases"]
    if case_type not in valid:
        raise ValueError(f"case_type: {case_type} must be one of the valid choices {valid}")
    if input_concrete is None:
        input_concrete = expected_string
    if isinstance(input_concrete, str):
        input_concrete = os.path.join(test_data_dir, input_concrete)
    if isinstance(input_concrete, os.PathLike):
        if case_type in ["CH4", "N2O"]:
            input_concrete = input_handler.read_natural_emissions(input_concrete, 'CH4', endyear=nyend)
        elif case_type == "emis":
            ih = input_handler.InputHandler({"nyend": nyend, "nystart": nystart, "emstart": nyend})
            input_concrete = ih.read_emissions(input_concrete)
            input_concrete.rename(columns={"CO2": "CO2_FF", "CO2.1": "CO2_AFOLU"}, inplace=True)
        elif case_type == "gases":
            input_concrete = input_handler.read_inputfile(input_concrete, True, year_end=nyend)
        else:
            input_concrete = input_handler.read_components(input_concrete)
    if not isinstance(input_concrete, pd.DataFrame):
        raise TypeError(f"input_concrete for {case_type} must be either a str, a path or a DataFrame")
    return input_concrete   
    


def define_scendata_for_scm(
        test_data_dir, 
        gaspam = None, 
        df_nat_ch4 = None, 
        df_nat_n2o = None, 
        df_conc = None, 
        df_emis = None, 
        nyend = 2023,
        nystart=1750,
        emstart = 1850,
        ):

    gaspam = get_df_from_input_w_data_handler(
        gaspam, 
        test_data_dir, 
        'gases_vupdate_2024_WMO_added_new.txt',
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="CH4"
    )

    # CH4-block
    df_nat_ch4 = get_df_from_input_w_data_handler(
        df_nat_ch4, 
        test_data_dir, 
        '/natemis_CH4_ode_method_from_Sep2025_updates.txt',
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="CH4"
    )
    df_nat_n2o = get_df_from_input_w_data_handler(
        df_nat_n2o, 
        test_data_dir, 
        '/natemis_N2O_ode_method_from_Sep2025_updates.txt',
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="N2O"
    )
    df_conc = get_df_from_input_w_data_handler(
        df_conc,
        test_data_dir,
        '/igcc_historical_conc_gases_vupdate_2024_WMO_added_new.txt',
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="conc"
    )
    df_emis = get_df_from_input_w_data_handler(
        df_emis,
        test_data_dir,
        '/historical_em_gases_vupdate_2024_WMO_added_new.txt',
        nyend=nyend,
        nystart=nystart,
        emstart=emstart,
        case_type="conc"
    )
    scenariodata = [{
            "gaspam_data": gaspam,
            "emstart": 1850,  
            "conc_run":False,
            "nystart": 1750,
            "nyend": nyend,
            "concentrations_data": df_conc,
            "emissions_data": df_emis,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm":24,
            "scenname" : "ssp245-short"
        }]
    return scenariodata


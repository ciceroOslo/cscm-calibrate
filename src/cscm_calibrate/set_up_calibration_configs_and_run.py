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

def get_df_from_input_w_data_handler(input_concrete, test_data_dir, expected_string, nyend=2023, case_type = "CH4"):
    valid = ["CH4", "N2O", "emis", "conc"]
    if case_type not in valid:
        raise ValueError(f"case_type: {case_type} must be one of the valid choices {valid}")
    if input_concrete is None:
        input_concrete = expected_string
    if isinstance(input_concrete, str):
        input_concrete = os.path.join(test_data_dir, input_concrete)
    if isinstance(input_concrete, os.PathLike):
        if case_type == "CH4":
            input_concrete = input_handler.read_natural_emissions()
    


def define_scendata_for_scm(
        test_data_dir=None, 
        gases_name = 'gases_vupdate_2024_WMO_added_new.txt', 
        df_nat_ch4 = None, 
        df_nat_n2o = None, 
        df_conc = None, 
        emis_files = None, 
        nyend = 2023,

        ):

    gaspam = input_handler.read_components(os.path.join(test_data_dir, gases_name))

    # CH4-block
    if df_nat_ch4 is None:
        df_nat_ch4 = '/natemis_CH4_ode_method_from_Sep2025_updates.txt'
    if isinstance(df_nat_ch4, str):
        df_nat_ch4 = input_handler.read_natural_emissions(os.path.join(test_data_dir, df_nat_ch4) ,'CH4', endyear=nyend)
    elif isinstance(df_nat_ch4, os.PathLike):
        df_nat_ch4 = input_handler.read_natural_emissions(df_nat_ch4, 'CH4', endyear=nyend)
    if not isinstance(df_nat_ch4, pd.DataFrame):
        raise TypeError("df_nat_ch4 must be either a str, a path or a DataFrame")
    
    # N2O-block
    if df_nat_n2o is None:
        df_nat_n2o = '/natemis_N2O_ode_method_from_Sep2025_updates.txt'
    if isinstance(df_nat_ch4, str):
        df_nat_n2o = input_handler.read_natural_emissions(os.path.join(test_data_dir, df_nat_n2o) ,'N2O', endyear=nyend)
    elif isinstance(df_nat_n2o, os.PathLike):
        df_nat_n2o = input_handler.read_natural_emissions(df_nat_n2o, 'N2O', endyear=nyend)
    if not isinstance(df_nat_n2o, pd.DataFrame):
        raise TypeError("df_nat_n2o must be either a str, a path or a DataFrame")
    
    # Conc-block
    if df_conc is None:
        df_conc = '/igcc_historical_conc_gases_vupdate_2024_WMO_added_new.txt'
    if isinstance(df_nat_ch4, str):
        df_conc = os.path.join(test_data_dir, df_conc)
        df_ssp2_conc =input_handler.read_inputfile(test_data_dir + , True, year_end=nyend)
    elif isinstance(df_nat_n2o, os.PathLike):
        df_nat_n2o = input_handler.read_natural_emissions(df_nat_n2o, 'N2O', endyear=nyend)
    if not isinstance(df_nat_n2o, pd.DataFrame):
        raise TypeError("df_nat_n2o must be either a str, a path or a DataFrame")  

    df_ssp2_conc =input_handler.read_inputfile(test_data_dir + '/igcc_historical_conc_gases_vupdate_2024_WMO_added_new.txt', True, year_end=nyend)

    ih = input_handler.InputHandler({"nyend": nyend, "nystart": 1750, "emstart": 1850})
    emi_input =ih.read_emissions(test_data_dir + '/historical_em_gases_vupdate_2024_WMO_added_new.txt')
    emi_input.rename(columns={"CO2": "CO2_FF", "CO2.1": "CO2_AFOLU"}, inplace=True)


    scenariodata = [{
            "gaspam_data": gaspam,
            "emstart": 1850,  
            "conc_run":False,
            "nystart": 1750,
            "nyend": nyend,
            "concentrations_data": df_ssp2_conc,
            "emissions_data": emi_input,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm":24,
            "scenname" : "ssp245-short"
        }]
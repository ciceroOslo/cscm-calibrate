import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

from ciceroscm.parallel.distributionrun import DistributionRun

from run_full_rcmip_protocol import read_output_variables_from_protocol, make_scenariodata_argdict, make_dataframe_of_zeros

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))
from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm


INPUT_DIR = "/div/no-backup-nac/users/masan/GRAFITE/temp_indata/"
CONFIG_NAME = "draw_samples_no_efficacy_no_pattern_wide_lambda_400"
CONFIG_PATH = Path(f"../draw_samples_archive/{CONFIG_NAME}.json")


INPUT_FILENAME = f"1pctCO2_rcmip_{CONFIG_NAME}.csv"
EXCEEDANCE_THRESHOLDS_PG_C = [750, 1000, 2000]
#INPUT_DIR = "/home/masan/temp/rcmip_inputs_cscm/"

OUTPATH_MAIN = "out_file_dump_nopattern_noefficacy"

distrorun = DistributionRun(None, json_file_name=CONFIG_PATH)
arg_dict_piControl = make_scenariodata_argdict("idealised", "esm-piControl", "piControl", yend=2500)

scendata_piControl = define_scendata_for_scm(INPUT_DIR, **arg_dict_piControl)

def get_co2_trajectory_from_scenario(scenario_name):
    """
    Get the CO2 trajectory from the scenario.
    """
    fpath = os.path.join(INPUT_DIR, f"{scenario_name}_em_gases_vupdate_2024_WMO_added_new.txt")
    df = pd.read_csv(fpath, sep='\s+', index_col=0, skiprows=[1,2,3])
    co2_ts_full = df["CO2"].values
    return co2_ts_full

def make_scenariodata_from_co2_scen_with_branch(branch_year, scenario, co2_ts_full):
    """
    Make a dictionary of scenario data for idealised with scne-7 CO2 to branch year.
    """
    scendata = scendata_piControl.copy()
    scendata[0]['scenname'] = f"esm-{scenario}-brch-{branch_year}"
    scendata[0]['emstart'] = 2501
    co2_ts = np.zeros(len(co2_ts_full))
    co2_ts[:branch_year-1750] = co2_ts_full[:branch_year-1750]
    scendata[0]['emissions_data']['CO2_FF'] = co2_ts
    return scendata

branch_years = np.arange(1950, 2151, 5)

scenarios = ["scen7-H", "scen7-M", "scen7-ML"]

variables_list = read_output_variables_from_protocol(os.path.join(INPUT_DIR, "rcmip_phase3_protocol_v1.1.0.xlsx"))
variables_list_to_use = variables_list[0] + ["Emissions|CO2"]

done_hist = False
for scenario in scenarios:
    co2_ts_full = get_co2_trajectory_from_scenario(scenario)
    for branch_year in branch_years:
        if branch_year <= 2020 and done_hist:
            continue
        if branch_year > 2020:
            done_hist = True
        scendata = make_scenariodata_from_co2_scen_with_branch(branch_year, scenario, co2_ts_full)
        results = distrorun.run_over_distribution(
            scendata, output_vars = variables_list_to_use, max_workers = 20,
        )
        outpath = os.path.join(OUTPATH_MAIN, "Extras_For_Norman", f"esm-{scenario}-brch-{branch_year}.csv")
        if not results.empty:
            results.to_csv(outpath)
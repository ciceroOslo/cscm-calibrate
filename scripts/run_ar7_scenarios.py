import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

from ciceroscm.parallel.distributionrun import DistributionRun

from run_full_rcmip_protocol import read_output_variables_from_protocol, make_scenariodata_argdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))
from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm


INPUT_DIR = "/div/no-backup-nac/users/masan/GRAFITE/temp_indata/"
CONFIG_NAME = "draw_samples_pattern_pdo_ar7_500"
CONFIG_PATH = Path(f"../draw_samples_archive/{CONFIG_NAME}.json")

OUTPATH_MAIN = "out_file_dump_pdo_ar7"

distrorun = DistributionRun(None, json_file_name=CONFIG_PATH)

experiment_list = {
    "abrupt-4xCO2": [1750, 2000],
    "1pctCO2": [1750, 1920],
    "esm-flat10": [1750, 2070],
    "esm-allGHG-hist": [1750, 2023],
    "esm-flat10-zec": [1750, 2070],
    "esm-flat10-cdr": [1750, 2070],
    "esm-piControl": [1750, 2070],
    "esm-allGHG-scen7-VL": [1750, 2500],
    "esm-allGHG-scen7-L": [1750, 2500],
    "esm-allGHG-scen7-M": [1750, 2500],
    "esm-allGHG-scen7-LN": [1750, 2500],
    "esm-allGHG-scen7-ML": [1750, 2500],
    "esm-allGHG-scen7-HL": [1750, 2500],
    "esm-allGHG-scen7-H": [1750, 2500],
    "esm-allGHG-ssp2-com":  [1750, 2100],
}

variables_list = read_output_variables_from_protocol(os.path.join(INPUT_DIR, "rcmip_phase3_protocol_v1.1.0.xlsx"))

def take_scenario_row_define_scendata_and_run(experiment, forc_type, yend, ystart, dont_run=False, include_pdo = False):
    print(experiment)
    scen_name = experiment
    scen_name_strip = scen_name.split("esm-")[-1].split("allGHG-")[-1]
    if not scen_name.startswith("esm-allGHG"):
        run_type = "idealised"
        if scen_name.startswith("esm-"):
            variables = variables_list[0]
        else:
            variables = variables_list[0] + ["Emissions|CO2"]
    else:
        run_type = "non-idealised"
        variables = variables_list[2] + ["Anomalous Radiation","Dynamic Feedback Lambda"]
    if scen_name_strip.startswith("scen7") and scen_name_strip.endswith("C"):
        scen_name_strip = scen_name_strip[:-1]
    arg_dict = make_scenariodata_argdict(run_type, scen_name, scen_name_strip, yend)
    if forc_type == "nonatforc":
        arg_dict["sunvolc"] = 0
    # print(arg_dict.keys())
    # print(arg_dict["rf_luc_file"])
    #sys.exit(4)
    # TODO: Deal with natural emissions of ch4 and n2
    # Run conc, esm or esm-allghg
    #print(arg_dict)
    
    if run_type == "non-idealised" and include_pdo:
        pdo_timeseries = os.path.join(INPUT_DIR, "pdo_ts_noaa.dat")
        arg_dict["pdo_timeseries"] = pdo_timeseries

    scendata = define_scendata_for_scm(INPUT_DIR,**arg_dict)
    scendata[0]["scenname"] = scen_name

    if not scen_name.startswith("esm"):
        scendata[0]["conc_run"] = True
    if dont_run:
        with open("scenarios_inspected.txt", "a") as f:
            f.write(f"{scen_name}, {yend}, {ystart}, {scendata[0]['conc_run']}, {len(scendata)}\n")
            print(experiment)
            print(scen_name)
            print(scen_name_strip)
            print(arg_dict)
        if scen_name == "esm-hist":
            print(experiment)
            print(scen_name)
            print(scen_name_strip)
            print(arg_dict)
            sys.exit(4)
        return pd.DataFrame()


    

    # if row["Type"] == "idealised" or "piControl" in scen_name:
    #     return pd.DataFrame()


    #print(variables)
    # print(scen_name)
    # #print(row["Type"])
    # print(scendata[0]["emstart"])
    # print(scendata)
    #sys.exit(4)
    #try:
    results = distrorun.run_over_distribution(
        scendata, output_vars=variables, max_workers=20
    )
    # except Exception as e:
    #     print(f"Error running scenario {scen_name}: {e}")
    #     del scendata
    #     return pd.DataFrame()
    # Figure out what variables to output
    # Run over with distrorun, save outputs and move on

    # Special hook for branch runs etc...

    #
    del scendata
    return results

skip_for_now = ["esm-allGHG-ssp2-com"]

def main():
    for experiment, duration in experiment_list.items():
        if (skip_for_now and experiment in skip_for_now):# or not experiment.startswith("esm-allGHG"):
            print(f"Skipping {experiment} for now")
            continue
        outpath_natforc = f"{OUTPATH_MAIN}/{experiment}_ar7_natforc_{CONFIG_NAME}.csv"
        outpath_noforc = f"{OUTPATH_MAIN}/{experiment}_ar7_nonatforc_{CONFIG_NAME}.csv"
        if os.path.exists(outpath_natforc):
            print(f"Output file {outpath_natforc} already exists, skipping run")
            continue
        results = take_scenario_row_define_scendata_and_run(experiment, forc_type="natforc", ystart=duration[0], yend=duration[1], include_pdo = True)
        if not results.empty:
            results.to_csv(outpath_natforc)
        del results
        if experiment.startswith("esm-allGHG"):
            results = take_scenario_row_define_scendata_and_run(experiment, forc_type="nonatforc", ystart=duration[0], yend=duration[1], include_pdo = True)
            if not results.empty:
                results.to_csv(outpath_noforc)
            del results

if __name__ == "__main__":
    main()
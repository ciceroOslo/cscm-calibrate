import sys
import os
from pathlib import Path

import pandas as pd

from ciceroscm.parallel.distributionrun import DistributionRun

from run_full_rcmip_protocol import read_output_variables_from_protocol, make_scenariodata_argdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))
from cscm_calibrate.set_up_calibration_configs_and_run import (
    define_scendata_for_scm,
    get_df_from_input_w_data_handler,
)


INPUT_DIR = Path("/div/no-backup-nac/users/masan/GRAFITE/temp_indata/")
CONFIG_NAME = "draw_samples_no_efficacy_no_pattern_wide_lambda_400"
CONFIG_PATH = Path(f"../draw_samples_archive/{CONFIG_NAME}.json")

OUTPATH_MAIN = "out_file_dump_nopattern_noefficacy"
YSTART = 1750
EMSTART = 1850
MAX_YEND = 2500
DIAMOND_FILE_SUFFIX = "_em_gases_vupdate_2024_WMO_added_new_em_DiamondWP-scenarios.txt"
CONCENTRATIONS_FILE = "ssp245_conc_gases_vupdate_2024_WMO_added_new.txt"

distrorun = DistributionRun(None, json_file_name=CONFIG_PATH)
variables_list = read_output_variables_from_protocol(INPUT_DIR / "rcmip_phase3_protocol_v1.1.0.xlsx")


def get_diamond_emissions_files():
    return sorted(INPUT_DIR.glob(f"*{DIAMOND_FILE_SUFFIX}"))


def make_experiment_name(emissions_file):
    scenario_name = emissions_file.name.removesuffix(DIAMOND_FILE_SUFFIX)
    return f"esm-allGHG-{scenario_name}"


def get_emissions_end_year(emissions_file):
    emissions_data = get_df_from_input_w_data_handler(
        emissions_file.name,
        str(INPUT_DIR),
        emissions_file.name,
        nyend=MAX_YEND,
        nystart=YSTART,
        emstart=EMSTART,
        case_type="emis",
    )
    return int(emissions_data.index.max())


def take_scenario_row_define_scendata_and_run(experiment, emissions_file, yend, ystart, dont_run=False):
    print(experiment)
    scen_name = experiment
    scen_name_strip = scen_name.split("esm-")[-1].split("allGHG-")[-1]
    run_type = "non-idealised"
    variables = variables_list[0]
    if scen_name_strip.startswith("scen7") and scen_name_strip.endswith("C"):
        scen_name_strip = scen_name_strip[:-1]
    print(scen_name_strip)
    arg_dict = make_scenariodata_argdict(run_type, "esm-allGHG-ssp245", "ssp245", yend)
    arg_dict["df_emis"] = emissions_file
    arg_dict["df_conc"] = CONCENTRATIONS_FILE
    arg_dict["emstart"] = EMSTART
    arg_dict["sunvolc"] = 0

    scendata = define_scendata_for_scm(str(INPUT_DIR), **arg_dict)
    scendata[0]["scenname"] = scen_name
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


    

    results = distrorun.run_over_distribution(
        scendata, output_vars=variables, max_workers=20
    )
    del scendata
    return results

def main():
    diamond_emissions_files = get_diamond_emissions_files()
    if not diamond_emissions_files:
        raise FileNotFoundError(
            f"No Diamond emissions files matching '*{DIAMOND_FILE_SUFFIX}' found in {INPUT_DIR}"
        )

    for emissions_file in diamond_emissions_files:
        if "clews" in emissions_file.name:
            print(f"Skipping {emissions_file.name} as it contains 'clews'")
            continue

        experiment = make_experiment_name(emissions_file)
        outpath = f"{OUTPATH_MAIN}/{experiment}_idealised_{CONFIG_NAME}.csv"
        # if os.path.exists(outpath):
        #     print(f"Output file {outpath} already exists, skipping run")
        #     continue
        yend = get_emissions_end_year(emissions_file)
        print("Ready to run experiment:", experiment)
        print("Emissions file:", emissions_file)
        print(yend)
        results = take_scenario_row_define_scendata_and_run(
            experiment,
            emissions_file,
            ystart=YSTART,
            yend=yend,
        )
        if not results.empty:
            results.to_csv(outpath)
        del results

if __name__ == "__main__":
    main()
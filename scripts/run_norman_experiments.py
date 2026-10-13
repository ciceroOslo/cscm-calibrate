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

scenarios_lin = [f'esm-flat{X}-zec-{N}GtC-lin' for X in [1.25, 2.5, 5, 10, 20, 40] for N in [200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000]]
scenarios_acc = [f'esm-flat{X}-zec-{N}GtC-acc' for X in [1.25, 2.5, 5, 10, 20, 40] for N in [200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000]]
scenarios_dec = [f'esm-flat{X}-zec-{N}GtC-dec' for X in [1.25, 2.5, 5, 10, 20, 40] for N in [200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000]]

distrorun = DistributionRun(None, json_file_name=CONFIG_PATH)
arg_dict_piControl = make_scenariodata_argdict("idealised", "esm-piControl", "piControl", yend=2500)

scendata_piControl = define_scendata_for_scm(INPUT_DIR, **arg_dict_piControl)

def make_scenariodata_for_flat_extended_emissions(co2_ts, shape='lin', run_length=5000):
    """
    Make a dictionary of scenario data for flat extended emissions scenarios.
    """
    scendata = scendata_piControl.copy()
    scendata[0]['scenname'] = f"esm-flat{co2_ts[0]}-zec-{int(np.sum(co2_ts)/1e3)}GtC-{shape}"
    scenend = scendata[0]['nystart'] + run_length - 1
    scendata[0]['nyend'] = scenend
    scendata[0]['emstart'] = scenend + 1

    df_conc_stretch = pd.DataFrame(
        data=scendata[0]['concentrations_data'].tail(1).values.repeat(run_length, axis=0),
        index=pd.Index(
            data = np.arange(scendata[0]["nystart"], scenend+1),
            name = "Component"
        ),
        columns=scendata[0]['concentrations_data'].columns,
    )
    scendata[0]['concentrations_data'] = df_conc_stretch
    df_emis_stretch = pd.DataFrame(
        data=scendata[0]['emissions_data'].tail(1).values.repeat(run_length, axis=0),
        columns=scendata[0]['emissions_data'].columns,
        index=np.arange(scendata[0]["nystart"], scenend+1),
    )
    df_emis_stretch['CO2_FF'] = co2_ts
    scendata[0]['emissions_data'] = df_emis_stretch
    #sys.exit(4)
    print(scendata[0].keys())
    scendata[0]['rf_luc_data'] = make_dataframe_of_zeros("LUC", 0, run_length +1)
    #sys.exit(4)
    return scendata



def find_length_and_co2_trajectories(rate, cumulative, run_length_post_cessation=1000):
    """
    Find the length of the emission period and the CO2 trajectories for a given emission rate and cumulative emissions.
    """
    nyears = int(np.float64(cumulative)/np.float64(rate))
    nyears_tot = nyears + run_length_post_cessation
    co2_trajectory_flat = np.zeros(nyears_tot)
    co2_trajectory_acc = np.zeros(nyears_tot)
    co2_trajectory_dec = np.zeros(nyears_tot)
    co2_trajectory_flat[:nyears] = np.float64(cumulative)/nyears
    alpha = 2.5  # steepness
    t = np.arange(1, nyears + 1, dtype=float)
    shape_acc = t ** alpha
    co2_trajectory_acc[:nyears] = shape_acc * (np.float64(cumulative) / shape_acc.sum())
    shape_dec = (nyears + 1 - t ) ** alpha
    co2_trajectory_dec[:nyears] = shape_dec * (np.float64(cumulative) / shape_dec.sum())
    return nyears_tot, [co2_trajectory_flat, co2_trajectory_acc, co2_trajectory_dec]



Erates = [1.25, 2.5, 5, 10, 20, 40]
Ecum = [200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000]

# scen7_H = [f'esm-scen7-H-brch{X}' for X in np.arange(1950,2151,5)]
# scen7_M = [f'esm-scen7-M-brch{X}' for X in np.arange(1950,2151,5)]
# scen7_ML = [f'esm-scen7-ML-brch{X}' for X in np.arange(1950,2151,5)]

E_scenarios_lin = np.zeros([len(Erates),len(Ecum),5000])
E_scenarios_acc = np.zeros([len(Erates),len(Ecum),5000])
E_scenarios_dec = np.zeros([len(Erates),len(Ecum),5000])
shapes = ["lin", "acc", "dec"]
variables_list = read_output_variables_from_protocol(os.path.join(INPUT_DIR, "rcmip_phase3_protocol_v1.1.0.xlsx"))
variables_list_to_use = variables_list[0] + ["Emissions|CO2"]

for rate in Erates:
    for cumulative in Ecum:
        nyears_tot, trajectories = find_length_and_co2_trajectories(rate, cumulative)
        for s, shape in enumerate(shapes):
            scendata = make_scenariodata_for_flat_extended_emissions(trajectories[s], shape, run_length=nyears_tot)
            results = distrorun.run_over_distribution(
                scendata, output_vars = variables_list_to_use, max_workers = 20,
            )
            outpath = os.path.join(OUTPATH_MAIN, "Extras_For_Norman", f"esm-flat{rate}-shape-{cumulative}GtC-{shape}.csv")
            if not results.empty:
                results.to_csv(outpath)
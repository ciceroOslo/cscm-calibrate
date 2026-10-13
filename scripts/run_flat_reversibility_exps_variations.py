import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

rates = [ 0.75, 1.25, 2.5, 5, 10, 15]

nzero_years = [0, 10, 50, 100]
cutoff_cumulative = [200, 400, 600, 800, 1000]

scenarios_cdr = [f'esm-flat{X}-cdr-{N}GtC-nz-{Y}years' for X in rates for N in cutoff_cumulative for Y in nzero_years]
scenarios_rev = [f'esm-flat{X}-rev-{N}GtC-acc-{Y}years' for X in rates for N in cutoff_cumulative for Y in nzero_years]

distrorun = DistributionRun(None, json_file_name=CONFIG_PATH)
arg_dict_piControl = make_scenariodata_argdict("idealised", "esm-piControl", "piControl", yend=2500)

scendata_piControl = define_scendata_for_scm(INPUT_DIR, **arg_dict_piControl)

def make_scenariodata_for_flat_extended_emissions(co2_ts, nz_years, type='cdr', run_length=5000):
    """
    Make a dictionary of scenario data for flat extended emissions scenarios.
    """
    scendata = scendata_piControl.copy()
    scendata[0]['scenname'] = f"esm-flat{co2_ts[0]}-{type}-{int(np.sum(co2_ts)/1e3)}GtC-{nz_years}years"
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



def find_length_and_co2_trajectories(rate, cumulative, nz_years, run_length_post_cessation=500):
    """
    Find the length of the emission period and the CO2 trajectories for a given emission rate and cumulative emissions.
    """
    nyears_to_max = int(np.float64(cumulative)/np.float64(rate))
    nyears_to_max_cdr = int(np.float64(cumulative-rate*nz_years/4)/np.float64(rate))
    nyears_tot = nyears_to_max*2 + nz_years
    co2_trajectory_cdr = np.zeros(nyears_tot + run_length_post_cessation)
    co2_trajectory_rev = np.zeros(nyears_tot + run_length_post_cessation)
    co2_trajectory_cdr[:nyears_to_max_cdr] = np.float64(cumulative-rate*nz_years/4)/nyears_to_max
    print(nz_years)
    print(np.linspace(rate,-rate,nz_years))
    print(rate)
    print(cumulative)
    print(nyears_to_max_cdr)
    print(co2_trajectory_cdr[nyears_to_max_cdr:nyears_to_max_cdr+nz_years])
    co2_trajectory_cdr[nyears_to_max_cdr:nyears_to_max_cdr+nz_years] = np.linspace(rate,-rate,nz_years)
    co2_trajectory_cdr[nyears_to_max_cdr+nz_years:nyears_tot] = -rate
    co2_trajectory_rev[:nyears_to_max] = np.float64(cumulative)/nyears_to_max
    co2_trajectory_rev[nyears_to_max:nyears_to_max+nz_years] = 0
    co2_trajectory_rev[nyears_to_max+nz_years:nyears_tot] = -rate


    t = np.arange(1, nyears_tot + run_length_post_cessation + 1, dtype=float)
    return nyears_tot + run_length_post_cessation, [co2_trajectory_cdr, co2_trajectory_rev]



variables_list = read_output_variables_from_protocol(os.path.join(INPUT_DIR, "rcmip_phase3_protocol_v1.1.0.xlsx"))
variables_list_to_use = variables_list[0] + ["Emissions|CO2"]

fig1, axs1 = plt.subplots(1, 2, figsize=(15, 10))
fig2, axs2 = plt.subplots(1, 2, figsize=(15, 10))

colours_rates = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
ls_cum = ["-", "--", "-.", ":"]
for i, rate in enumerate(rates):
    for j, cumulative in enumerate(cutoff_cumulative):
        for nz_years in nzero_years:
            if int(np.float64(cumulative-rate*nz_years/4)/np.float64(rate)) < 0:
                print(f"Meaningful trajectory not possible for rate {rate} GtC/y, cumulative {cumulative} GtC, nz_years {nz_years} combination")
                continue
            nyears_tot, trajectories = find_length_and_co2_trajectories(rate, cumulative, nz_years, run_length_post_cessation=100)
            axs1[0].plot(np.arange(1, nyears_tot + 1), trajectories[0], color=colours_rates[i], ls= ls_cum[j%len(ls_cum)])
            axs1[1].plot(np.arange(1, nyears_tot + 1), trajectories[1], color = colours_rates[i], ls= ls_cum[j%len(ls_cum)])
            axs2[0].plot(np.arange(1, nyears_tot + 1), np.cumsum(trajectories[0]), color=colours_rates[i], ls= ls_cum[j%len(ls_cum)])
            axs2[1].plot(np.arange(1, nyears_tot + 1), np.cumsum(trajectories[1]), color=colours_rates[i], ls= ls_cum[j%len(ls_cum)])
            for t, typescen in enumerate(["cdr", "rev"]):
                outpath = os.path.join(OUTPATH_MAIN, "Extras_For_Norman", f"esm-flat{rate}-{typescen}-{cumulative}GtC-{nz_years}years.csv")
                if os.path.exists(outpath):
                    print(f"File {outpath} already exists, skipping")
                    continue
                scendata = make_scenariodata_for_flat_extended_emissions(trajectories[t], typescen, run_length=nyears_tot)
                try:
                    results = distrorun.run_over_distribution(
                        scendata, output_vars = variables_list_to_use, max_workers = 20,
                    )
                except Exception as e:
                    print(f"Error running distribution for {outpath}: {e}")
                    continue
                if not results.empty:
                    results.to_csv(outpath)
axs1[0].set_title("CO2 Trajectory with CDR")
axs1[1].set_title("CO2 Trajectory with Reversibility")
axs2[0].set_title("CO2 Trajectory with CDR")
axs2[1].set_title("CO2 Trajectory with Reversibility")
for num, ax in enumerate([axs1[0], axs1[1], axs2[0], axs2[1]]):
    ax.set_xlabel("Years")
    if num < 2:
        ax.set_ylabel("CO2 GtC/yr")
    else:
        ax.set_ylabel("CO2 Cumulative GtC")
fig1.suptitle("CO2 Trajectories for Different Scenarios")
fig2.suptitle("CO2 Cumulative Trajectories for Different Scenarios")
fig1.savefig("co2_idealised_emissions.png")
fig2.savefig("co2_idealised_cumulative_emissions.png")
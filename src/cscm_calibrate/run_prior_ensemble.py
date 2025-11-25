#!/usr/bin/env python

# # CICERO SCM notebook - parallel application

# Import some stuff

# In[1]:


import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    from pandas.core.common import SettingWithCopyWarning
except:
    from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_project_root():
    """Get the project root directory (where this package is installed)."""
    # Go up from src/cscm_calibrate/ to the project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


warnings.filterwarnings("ignore", message=".*Parameter.*")

cscm_path = cscm_path = (
    "/home/masan/gitrepos/ciceroscm"  # os.path.join("..", "..", "..", "ciceroscm")
)


sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.parallel.distributionrun import DistributionRun


def run_prior_ensemble(
    testconfig,
    scenariodata,
    calibdata,
    prunecfgs,
    distnums=6000000,
    chunk_size=10000,
    startdate=None,
):
    """
    Run a prior ensemble simulation, processes results, and saves outputs for calibration.

    This function generates configuration lists, runs simulations over a distribution of parameter sets,
    processes the results for calibration, and saves the relevant outputs and parameter matrices to disk.
    The simulation is performed in chunks to handle large numbers of distributions efficiently.

    Parameters
    ----------
    testconfig : object
        Configuration object with methods for generating configuration lists.
    scenariodata : object or DataFrame
        Scenario data required for running the distribution simulations.
    calibdata : pandas.DataFrame
        DataFrame containing calibration variable information and year ranges.
    prunecfgs : dict
        Dictionary mapping variable names to configuration information for pruning and saving results.
    distnums : int, optional
        Total number of distributions to simulate (default is 6,000,000).
    chunk_size : int, optional
        Number of distributions to process per chunk (default is 10,000).
    startdate : str or None, optional
        Optional string to append to output filenames for distinguishing runs (default is None).

    Returns
    -------
    None
        This function saves results to disk and does not return any value.

    Side Effects
    ------------
    - Saves numpy arrays and HDF5 files with simulation results and parameter matrices in the 'data/' directory.
    - May print debugging information if print statements are uncommented.

    Notes
    -----
    - The function assumes the existence of a `DistributionRun` class and specific structure in `calibdata`.
    - Output files are named according to the number of distributions, chunk index, and optional start date.
    """
    if startdate is None:
        startdate = ""

    # Ensure we save to project root output directory
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    testconfig.make_config_lists(
        distnums,
        json_fname=os.path.join(output_dir, f"configs_{distnums}_.json"),
        max_chunk_size=chunk_size,
    )
    chunk_nums = int(np.ceil(distnums / chunk_size))

    for i in range(chunk_nums):
        file_midstring = f"{distnums}_chunk_{i}{startdate}"
        distrorun1 = DistributionRun(
            testconfig,
            json_file_name=os.path.join(output_dir, f"configs_{file_midstring}.json"),
            numvalues=distnums,
        )
        output_vars = calibdata["Variable Name"]
        results = distrorun1.run_over_distribution(
            scenariodata, output_vars, max_workers=200
        )

        # print(results.head())
        # print(results.shape)

        results_for_fit_dict_1d = {}
        for idx, data in calibdata.iterrows():
            # print(data)
            results_sub = results.loc[results["variable"] == data["Variable Name"]]
            if (
                data["Yearstart_norm"] == data["Yearend_norm"]
                and data["Yearstart_norm"] == 1750
            ):
                results_for_fit_dict_1d[data["Variable Name"]] = results_sub.iloc[
                    :, data["Yearstart_change"] - 1750 + 7
                ].values
            elif data["Yearstart_norm"] == data["Yearend_norm"]:
                print(results_sub.iloc[:, data["Yearstart_change"] - 1750 + 7])
                results_for_fit_dict_1d[data["Variable Name"]] = (
                    results_sub.iloc[:, data["Yearstart_change"] - 1750 + 7]
                    - results_sub.iloc[:, data["Yearstart_norm"] - 1750 + 7]
                ).values
            else:
                results_for_fit_dict_1d[data["Variable Name"]] = (
                    (
                        results_sub.iloc[
                            :,
                            data["Yearstart_change"] - 1750 + 7 : data["Yearend_change"]
                            - 1750
                            + 8,
                        ]
                    ).mean(axis=1)
                    - (
                        results_sub.iloc[
                            :,
                            data["Yearstart_norm"] - 1750 + 7 : data["Yearend_norm"]
                            - 1750
                            + 8,
                        ]
                    ).mean(axis=1)
                ).values

        print(f"DEBUG: prunecfgs = {prunecfgs}")
        print(f"DEBUG: About to save .npy files for {len(prunecfgs)} variables")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Saving to output directory: {output_dir}")
        for variable, varinfo in prunecfgs.items():
            print(f"DEBUG: Processing variable '{variable}' with varinfo {varinfo}")
            results_save = results[results["variable"] == varinfo[0]]
            print(f"DEBUG: results_save shape before filtering: {results_save.shape}")
            ids = results_save["run_id"].to_numpy()
            results_save = results_save.iloc[:, 107:].to_numpy(float)
            filename = os.path.join(
                output_dir, f"{variable}_{file_midstring}_1850-2023.npy"
            )
            abs_filename = os.path.abspath(filename)
            print(f"DEBUG: Saving to {filename} with shape {results_save.shape}")
            np.save(filename, results_save)
            np.save(os.path.join(output_dir, f"sample_ids_{file_midstring}.npy"), ids)
            print(f"DEBUG: Files saved successfully")
            print(f"DEBUG: File exists check: {os.path.exists(abs_filename)}")
        # sys.exit(4)
        targ = pd.DataFrame(data=results_for_fit_dict_1d)
        targ.index.set_names("run_id", inplace=True)
        pdict = distrorun1.cfgs

        def merge_dicts(dc):
            x = dc["pamset_udm"]
            y = dc["pamset_emiconc"]
            w = dc["pamset_carbon"]
            z = x.copy()
            z.update(y)
            z.update(w)
            return z

        mdict = [merge_dicts(d) for d in pdict]
        pmat = pd.DataFrame(mdict)

        parammat = pmat.loc[:, (pmat != pmat.iloc[0]).any()]
        parammat

        # sys.exit(4)
        store = pd.HDFStore(os.path.join(output_dir, f"data_{file_midstring}.h5"))
        store["targ"] = targ
        store["parammat"] = parammat
        store.close()

        # store_long = pd.HDFStore('data/data_long.h5')
        # store_long['results'] = results
        # store_long.close()


# if __name__ == "__main__":
#     test_data_dir = os.path.join(
#         os.getcwd(), "../../", "data", "calibration_data_Sep2025"
#     )
#     gaspam = input_handler.read_components(
#         test_data_dir + "/gases_vupdate_2024_WMO_added_new.txt"
#     )
#     print(gaspam.head())
#     nyend = 2023

#     df_nat_ch4 = input_handler.read_natural_emissions(
#         test_data_dir + "/natemis_CH4_ode_method_from_Sep2025_updates.txt",
#         "CH4",
#         endyear=nyend,
#     )
#     df_nat_n2o = input_handler.read_natural_emissions(
#         test_data_dir + "/natemis_N2O_ode_method_from_Sep2025_updates.txt",
#         "N2O",
#         endyear=nyend,
#     )
#     print(df_nat_ch4.head())

#     df_ssp2_conc = input_handler.read_inputfile(
#         test_data_dir + "/igcc_historical_conc_gases_vupdate_2024_WMO_added_new.txt",
#         True,
#         year_end=nyend,
#     )

#     ih = input_handler.InputHandler({"nyend": nyend, "nystart": 1750, "emstart": 1850})
#     emi_input = ih.read_emissions(
#         test_data_dir + "/historical_em_gases_vupdate_2024_WMO_added_new.txt"
#     )
#     emi_input.rename(columns={"CO2": "CO2_FF", "CO2.1": "CO2_AFOLU"}, inplace=True)

#     prior_distro_dict = {
#         "rlamdo": [5, 25],
#         "akapa": [0.06, 0.8],
#         "cpi": [0.161, 0.569],
#         "W": [0.55, 2.55],
#         "beto": [0, 7],
#         "lambda": [2 / 3.71, 5 / 3.71],
#         "mixed": [25, 125],
#         "qo3": [0.4, 0.6],
#         "qdirso2": [-0.005, -0.000],
#         "qindso2": [-0.02, -0.00],
#         "qbc": [0.004, 0.05],
#         "qoc": [-0.008, -0.001],
#         "beta_f": [0.110, 1.0],
#         "mixed_carbon": [25, 125],
#         "solubility_sens": [0, 0.03],
#         "ocean_efficacy": [0.9, 1.3],
#         "ml_w_sigmoid": [2.0, 7.0],
#         "ml_fracmax": [0.0, 1.0],
#         "t_half": [0.3, 0.8],
#         "t_threshold": [3, 10],
#         "w_threshold": [2, 8],
#         "w_sigmoid": [2, 8],
#     }

#     testconfig = _ConfigDistro(
#         distro_dict=prior_distro_dict,
#         setvalues={
#             "threstemp": 7.0,
#             "lm": 40,
#             "ldtime": 12,
#             "qbmb": 0,
#             "solubility_limit": 0.1,
#             "ml_t_half": 0.0,
#         },
#     )

#     scenariodata = [
#         {
#             "gaspam_data": gaspam,
#             "emstart": 1850,
#             "conc_run": False,
#             "nystart": 1750,
#             "nyend": nyend,
#             "concentrations_data": df_ssp2_conc,
#             "emissions_data": emi_input,
#             "nat_ch4_data": df_nat_ch4,
#             "nat_n2o_data": df_nat_n2o,
#             "idtm": 24,
#             "scenname": "ssp245-short",
#         }
#     ]

#     calibdata = pd.DataFrame(
#         data={
#             "Variable Name": [
#                 "Heat Content|Ocean",
#                 "Surface Air Ocean Blended Temperature Change",
#                 "Effective Radiative Forcing|Aerosols",
#                 "Atmospheric Concentrations|CO2",
#                 "Ocean carbon flux",
#                 "Biosphere carbon flux",
#             ],
#             "Yearstart_norm": [1971, 1850, 1750, 1750, 1750, 1750],
#             "Yearend_norm": [1971, 1900, 1750, 1750, 1750, 1750],
#             "Yearstart_change": [2023, 2011, 2023, 2023, 2014, 2014],
#             "Yearend_change": [2023, 2020, 2023, 2023, 2023, 2023],
#             "Central Value": [484.82157000000063, 1.24, -1.18, 410.1 - 278, 2.9, 3.2],
#             "sigma": [36.8551891022091, 0.073, 0.7, 0.4, 0.4, 0.9],
#         }
#     )

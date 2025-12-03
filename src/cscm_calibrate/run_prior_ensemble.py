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

from .shared_functions import get_project_root
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


warnings.filterwarnings("ignore", message=".*Parameter.*")

# Get path to ciceroscm - one level up from project root
cscm_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ciceroscm")
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
    max_workers=200,
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
    max_workers : int, optional
        Number of parallel workers to use for ensemble runs (default is 200).

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
    print("Generating configuration lists...")
    # Suppress stdout from config generation to reduce noise
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:

        testconfig.make_config_lists(
            distnums,
            json_fname=os.path.join(output_dir, f"configs_{distnums}_.json"),
            max_chunk_size=chunk_size,
        )
        del testconfig # Free up memory
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    print("Configuration lists generated.")
    chunk_nums = int(np.ceil(distnums / chunk_size))
    
    print(f"\n{'='*60}")
    print(f"PRIOR ENSEMBLE GENERATION")
    print(f"{'='*60}")
    print(f"Total samples: {distnums:,}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Number of chunks: {chunk_nums}")
    print(f"Parallel workers: {max_workers}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for i in range(chunk_nums):
        print(f"\n--- Processing Chunk {i+1}/{chunk_nums} ---")
        file_midstring = f"{distnums}_chunk_{i}"
        print(os.path.join(output_dir, f"configs_{file_midstring}.json"))
        
        # Suppress stdout from DistributionRun initialization
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            distrorun1 = DistributionRun(
                None,
                json_file_name=os.path.join(output_dir, f"configs_{file_midstring}.json"),
                numvalues=distnums,
            )
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        
        output_vars = calibdata["Variable Name"]
        print(f"Running {chunk_size:,} simulations with {max_workers} workers...")
        results = distrorun1.run_over_distribution(
            scenariodata, output_vars, max_workers=max_workers
        )

        results_for_fit_dict_1d = {}
        for idx, data in calibdata.iterrows():
            results_sub = results.loc[results["variable"] == data["Variable Name"]]
            if (
                data["Yearstart_norm"] == data["Yearend_norm"]
                and data["Yearstart_norm"] == 1750
            ):
                results_for_fit_dict_1d[data["Variable Name"]] = results_sub.iloc[
                    :, data["Yearstart_change"] - 1750 + 7
                ].values
            elif data["Yearstart_norm"] == data["Yearend_norm"]:
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

        # Save timeseries output for pruning variables
        for variable, varinfo in prunecfgs.items():
            results_save = results[results["variable"] == varinfo[0]]
            ids = results_save["run_id"].to_numpy()
            results_save = results_save.iloc[:, 107:].to_numpy(float)
            filename = os.path.join(
                output_dir, f"{variable}_{file_midstring}_1850-2023.npy"
            )
            np.save(filename, results_save)
            np.save(os.path.join(output_dir, f"sample_ids_{file_midstring}.npy"), ids)

        # Save constraint targets and parameter matrices
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

        h5_file = os.path.join(output_dir, f"data_{file_midstring}.h5")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            store = pd.HDFStore(h5_file)
            store["targ"] = targ
            store["parammat"] = parammat
            store.close()

        print(f"✓ Chunk {i+1}/{chunk_nums} complete!\n")

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

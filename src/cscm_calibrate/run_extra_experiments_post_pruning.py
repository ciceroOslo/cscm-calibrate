import glob
import os
import sys

import numpy as np
import pandas as pd

from .shared_functions import get_project_root, make_config_distro_json, make_list_of_chunks_to_run, compute_ecs_gregory, store_h5_data

# Get path to ciceroscm - one level up from project root
cscm_path = os.path.abspath(os.path.join(get_project_root(), "..", "ciceroscm"))


sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.parallel.distributionrun import (  # noqa: E402
    DistributionRun,
)
def read_pruned_and_run_per_chunks(
    config,
    scenario_data,
    calibration_data,
    idealised_experiments,
    file_endstring,
    continue_from_existing=False,
):
    """
    Run additional experiments after pruning step in chunks.

    This method reads the pruned members from CSV files matching the specified pattern,
    and runs additional experiments for each chunk of pruned members. It utilizes the
    scenario data and calibration data for idealised experiments defined in the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing settings for the experiments.
    scenario_data : pd.DataFrame
        DataFrame containing scenario data for the experiments.
    calibration_data : pd.DataFrame
        DataFrame containing calibration data for the experiments.
    idealised_experiments : list of dict
        List of dictionaries, each defining an idealised experiment to run.
    pruned_members_file_pattern : str, optional
        Glob pattern to match pruned members CSV files (default is "pruned_members_*.csv").
    continue_from_existing : bool, optional
        Whether to continue from existing results if they exist (default is False).

    Returns
    -------
    None
        This function does not return anything. It runs the experiments and saves results as specified in the configuration.
    """
    output_dir = os.path.join(get_project_root(), "output")
    if not os.path.exists(output_dir):
        print("No existing output path, no pruning data, this can only be run post pruning")
        sys.exit(4)

    num_chunks = _read_pruned_and_make_configs_for_chunks(config, output_dir, chunk_size=10000, file_endstring=file_endstring, continue_from_existing=continue_from_existing)

    print("Finished making config files for pruned chunks, now running idealised experiments for each chunk...")

    to_run = make_list_of_chunks_to_run(output_dir,f"data_postpruned_idealised_experiments_chunk_*.h5" , num_chunks)  # Set sample_max to a reasonable number based on expected number of chunks
    print (f"Chunks to run: {to_run}")
    #sys.exit(4)
    for i in to_run:
        print(f"Running chunk {i} of idealised experiments post pruning...")
        print(os.path.join(output_dir, f"pruned_ensemble_chunk_{i}.json"))
        targ, pdict = run_single_chunk_idealised_experiments(
            scenariodata=scenario_data,
            max_workers=config["prior_configs"]["max_workers"],
            output_dir=output_dir,
            config_file=os.path.join(output_dir, f"pruned_ensemble_chunk_{i}.json"),
            calib_data=calibration_data,
            chunk_size=10000
        )
        print(targ)
        store_h5_data(pdict, targ, output_dir, f"postpruned_idealised_experiments_chunk_{i}")
        print(f"✓ Chunk {i + 1}/{num_chunks} complete!\n")

    combine_targ_paramat_for_weighting(num_chunks)
    

def combine_targ_paramat_for_weighting(num_chunks):
    output_dir = os.path.join(get_project_root(), "output")

    idealise_targ_list = []
    for i in range(num_chunks):
        targ_file = os.path.join(output_dir, f"data_postpruned_idealised_experiments_chunk_{i}.h5")
        store = pd.HDFStore(targ_file)
        targ_chunk = store["targ"]
        store.close()
        idealise_targ_list.append(targ_chunk)
    all_idealised_targs = pd.concat(idealise_targ_list)

    store = pd.HDFStore(
        os.path.join(output_dir, f"data_all_targs_paramats.h5")
    )
    parammat = store["parammat"]
    targ = store["targ"]
        
    store.close()
    print(targ.head())
    print(targ.shape)
    print(all_idealised_targs.shape)
    print(all_idealised_targs.head())
    all_idealised_targs.reset_index(drop=True, inplace=True)
    targ.reset_index(drop=True, inplace=True)
    combined_targ = pd.concat([targ, all_idealised_targs], axis=1)
    print(combined_targ.columns)
    print(combined_targ.head())
    store = pd.HDFStore(
        os.path.join(output_dir, f"data_all_targs_paramats_include_idealised.h5")
    )
    store["parammat"] = parammat
    store["targ"] = combined_targ
    store.close()
    print("Combined targs and parammat for weighting step saved to data_all_targs_paramats_include_idealised.h5")


def _read_pruned_and_make_configs_for_chunks(
    self,
    output_dir,
    chunk_size,
    file_endstring,
    continue_from_existing=False
):

    valid_indices = np.load(os.path.join(output_dir, f"valid_sample_ids_all_chunks{file_endstring}.npy"), allow_pickle=True)
    num_runs = len(valid_indices)
    num_chunks = (num_runs + chunk_size - 1) // chunk_size  # Calculate number of chunks needed
    if continue_from_existing:
        if os.path.exists(output_dir):
            if os.path.exists(
                os.path.join(
                    output_dir, f"pruned_ensemble_chunk_{num_chunks -1}.json"
                )
            ):
                print("Using preexisting config files")
                return num_chunks

    store = pd.HDFStore(
        os.path.join(output_dir, f"data_all_targs_paramats{file_endstring}.h5")
    )
    parammat = store["parammat"]

    store.close()

    print(parammat.shape)
    print(parammat.head())
    print(parammat.columns)
    num_runs = parammat.shape[0]
    print(f"Total number of pruned runs: {num_runs}")
    num_chunks = (num_runs + chunk_size - 1) // chunk_size  #
    print("Number of chunks to run: ", num_chunks)

    print(f"Number of valid runs after pruning: {len(valid_indices)}")
    print(valid_indices)
    for i in range(num_chunks):
        make_config_distro_json(
            parammat.iloc[i * chunk_size : (i + 1) * chunk_size].to_numpy().transpose(),
            parammat.columns,
            f"pruned_ensemble_chunk_{i}{file_endstring}.json",
            index_list=valid_indices[i * chunk_size : (i + 1) * chunk_size],
            output_dir=output_dir,
        )

    print(f"Number of chunks to run: {num_chunks} with chunk size {chunk_size}")
    return num_chunks
    
def run_single_chunk_idealised_experiments(
    scenariodata, 
    max_workers,
    output_dir,
    config_file,
    calib_data,
    output_vars = None,
    chunk_size=10000):  # noqa: PLR0913
    """
    Example function to run a single chunk of the prior ensemble for testing purposes.
    This version is set up to run idealised experiments with hardcoded parameters.
    """
    # This function can be implemented similarly to run_prior_ensemble but with hardcoded parameters for a single chunk.

    # Suppress stdout from DistributionRun initialization
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        distrorun1 = DistributionRun(
            None,
            json_file_name=os.path.join(
                output_dir, config_file
            ),
            numvalues=chunk_size, # used to be distnums but we want to run one chunk at a time here
        )
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    #print(distrorun1.cfgs)

    #sys.exit(4)
    if output_vars is None:
        output_vars = ["Surface Air Ocean Blended Temperature Change",
                       "Heat Content|Ocean"
                       ]
    for scenarios in scenariodata:
        if "ref_yr" in scenarios.keys():
            for i, cfg in enumerate(distrorun1.cfgs):
                distrorun1.cfgs[i]["pamset_emiconc"]["ref_yr"] = scenarios["ref_yr"]
        #print(scenarios.keys())
        #print(scenarios["scenname"])
        #print(scenarios["rf_luc_data"])
    print(f"Running {chunk_size:,} simulations with {max_workers} workers...")
    results = distrorun1.run_over_distribution(
        scenariodata, output_vars, max_workers=max_workers
    )
    targ_list = []


    for i, experiment in enumerate(calib_data["Experiments"]):
        results_sub = results.loc[results["scenario"] == experiment]
        targ_comp = calib_data["Varname_short"][i]
        if targ_comp == "ECS":
            targ_list.append(compute_ecs_gregory(results_sub, calib_data["Yearstart_change"][i], calib_data["Yearend_change"][i]))
        elif targ_comp == "TCR" or targ_comp == "TCRE":
            targ_np = results_sub.loc[results_sub["variable"]=="Surface Air Ocean Blended Temperature Change"][calib_data["Yearend_change"][i]].values - results_sub.loc[results_sub["variable"]=="Surface Air Ocean Blended Temperature Change"][calib_data["Yearstart_norm"][i]].values
            targ_np = pd.Series(targ_np, name=targ_comp)
            targ_np.index.set_names("run_id", inplace=True)
            targ_list.append(targ_np)

            
    return pd.concat(targ_list, axis=1), distrorun1.cfgs

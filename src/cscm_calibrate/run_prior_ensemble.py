"""
Module to run prior ensemble simulations and process results.
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    from pandas.core.common import SettingWithCopyWarning
except:  # noqa: E722
    from pandas.errors import SettingWithCopyWarning

from .shared_functions import get_project_root

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


warnings.filterwarnings("ignore", message=".*Parameter.*")

# Get path to ciceroscm - one level up from project root
cscm_path = os.path.abspath(get_project_root(), "..", "ciceroscm")


sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.parallel.distributionrun import (  # noqa: E402
    DistributionRun,
)


def _generate_prior_ensemble_parameters(
    testconfig, output_dir, distnums, chunk_size, continue_from_existing=False
):
    if continue_from_existing:
        if os.path.exists(output_dir):
            chunk_nums = int(np.ceil(distnums / chunk_size))
            if os.path.exists(
                os.path.join(
                    output_dir, f"configs_{distnums}_chunk_{chunk_nums - 1}.json"
                )
            ):
                print("Using preexisting config files")
                return
    os.makedirs(output_dir, exist_ok=True)
    print("Generating configuration lists...")
    # Suppress stdout from config generation to reduce noise
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        testconfig.make_config_lists(
            distnums,
            json_fname=os.path.join(output_dir, f"configs_{distnums}_.json"),
            max_chunk_size=chunk_size,
        )
        del testconfig  # Free up memory
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    print("Configuration lists generated.")

    return


def run_prior_ensemble(  # noqa: PLR0913, PLR0915
    testconfig,
    scenariodata,
    calibdata,
    prunecfgs,
    distnums=6000000,
    chunk_size=10000,
    startdate=None,
    max_workers=200,
    continue_from_existing=False,
):
    """
    Run a prior ensemble simulatio and process results

    This function generates configuration lists,
    runs simulations over a distribution of parameter sets,
    processes the results for calibration,
    and saves the relevant outputs and parameter matrices to disk.
    The simulation is performed in chunks to handle
    large numbers of distributions efficiently.

    Parameters
    ----------
    testconfig : object
        Configuration object with methods for generating configuration lists.
    scenariodata : object or DataFrame
        Scenario data required for running the distribution simulations.
    calibdata : pandas.DataFrame
        DataFrame containing calibration variable information and year ranges.
    prunecfgs : dict
        Dictionary mapping variable names to configuration information for
        pruning and saving results.
    distnums : int, optional
        Total number of distributions to simulate (default is 6,000,000).
    chunk_size : int, optional
        Number of distributions to process per chunk (default is 10,000).
    startdate : str or None, optional
        Optional string to append to output filenames for distinguishing runs
        (default is None).
    max_workers : int, optional
        Number of parallel workers to use for ensemble runs (default is 200).

    Returns
    -------
    None
        This function saves results to disk and does not return any value.

    Side Effects
    ------------
    - Saves numpy arrays and HDF5 files with simulation results
      and parameter matrices in the 'data/' directory.
    - May print debugging information if print statements are uncommented.

    Notes
    -----
    - The function assumes the existence of a `DistributionRun` class
      and specific structure in `calibdata`.
    - Output files are named according to the number of distributions,
      chunk index, and optional start date.
    """
    if startdate is None:
        startdate = ""

    # Ensure we save to project root output directory
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "output")
    _generate_prior_ensemble_parameters(
        testconfig=testconfig,
        output_dir=output_dir,
        distnums=distnums,
        chunk_size=chunk_size,
        continue_from_existing=continue_from_existing,
    )
    chunk_nums = int(np.ceil(distnums / chunk_size))
    sample_max = -1
    if continue_from_existing:
        sample_dumps_existing = glob.glob(
            f"{output_dir}/sample_ids_{distnums}_chunk_*.npy"
        )
        if len(sample_dumps_existing) > 0:
            sample_dumps_existing = [
                int(fpath.split("_")[-1].split(".")[0])
                for fpath in sample_dumps_existing
            ]
            sample_dumps_existing.sort()
            sample_max = sample_dumps_existing[-1]

    print(f"\n{'=' * 60}")
    print("PRIOR ENSEMBLE GENERATION")
    print(f"{'=' * 60}")
    print(f"Total samples: {distnums:,}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Number of chunks: {chunk_nums}")
    print(f"Parallel workers: {max_workers}")
    print(f"Output directory: {output_dir}")
    if continue_from_existing:
        print(f"Continuing from chunk {sample_max + 1}")
    print(f"{'=' * 60}\n")
    # sys.exit(4)

    for i in range(sample_max + 1, chunk_nums):
        print(f"\n--- Processing Chunk {i + 1}/{chunk_nums} ---")
        file_midstring = f"{distnums}_chunk_{i}"
        print(os.path.join(output_dir, f"configs_{file_midstring}.json"))

        # Suppress stdout from DistributionRun initialization
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            distrorun1 = DistributionRun(
                None,
                json_file_name=os.path.join(
                    output_dir, f"configs_{file_midstring}.json"
                ),
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
        syear = 1750
        results_for_fit_dict_1d = {}
        for idx, data in calibdata.iterrows():
            results_sub = results.loc[results["variable"] == data["Variable Name"]]
            if (
                data["Yearstart_norm"] == data["Yearend_norm"]
                and data["Yearstart_norm"] == syear
            ):
                results_for_fit_dict_1d[data["Variable Name"]] = results_sub.iloc[
                    :, data["Yearstart_change"] - syear + 7
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
                            data["Yearstart_change"]
                            - 1750
                            + 7 : data["Yearend_change"]
                            - 1750
                            + 8,
                        ]
                    ).mean(axis=1)
                    - (
                        results_sub.iloc[
                            :,
                            data["Yearstart_norm"]
                            - 1750
                            + 7 : data["Yearend_norm"]
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
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            store = pd.HDFStore(h5_file)
            store["targ"] = targ
            store["parammat"] = parammat
            store.close()

        print(f"✓ Chunk {i + 1}/{chunk_nums} complete!\n")

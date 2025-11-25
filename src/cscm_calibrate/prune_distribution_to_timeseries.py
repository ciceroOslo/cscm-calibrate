import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .shared_functions import rmse


def prepare_weights_temp(data_path, data_varname="GMST"):
    """
    Prepares weights and extracts a temperature time series from a CSV file.
    This function reads a CSV file containing temperature data, extracts the specified variable as a NumPy array,
    and generates a corresponding weights array for use in temperature pruning. The weights array is initialized
    with ones, except for the first element (set to 0.5) and the last element (set to 0.0).

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the temperature data.
    data_varname : str, optional
        Name of the column in the CSV file to extract as the temperature time series (default is "GMST").

    Returns
    -------
    gmst : numpy.ndarray
        Array containing the extracted temperature time series.
    weights : numpy.ndarray
        Array of weights for the temperature time series, with custom values at the first and last positions.
    """
    weights = np.ones(52)
    weights[0] = 0.5
    weights[-1] = 0.0
    # Temperature pruning input

    temp_data = pd.read_csv(data_path)
    gmst = temp_data[data_varname].values
    return gmst, weights


def do_pruning_for_chunk(
    chunk_num, prune_list, file_endstring=None, total_samples=6000000
):
    """
    Prunes a chunk of temperature timeseries samples based on RMSE constraints.
    Loads temperature samples and corresponding sample IDs for a given chunk, computes the RMSE
    between each sample's temperature timeseries and observed GMST (after baseline adjustment),
    and returns the indices and IDs of samples passing the RMSE threshold.

    Parameters
    ----------
    chunk_num : int
        The chunk number to process.
    prune_list : list
        A list containing:
            - str: The base filename for temperature data.
            - str: The filename or identifier for observed GMST data.
            - float: The RMSE acceptance threshold.
    file_endstring : str, optional
        Optional string to append to the sample IDs filename (default is None).
    total_samples : int, optional
        Total number of samples in the dataset (default is 6,000,000).

    Returns
    -------
    valid_temp : np.ndarray
        Array of indices of samples passing the RMSE constraint.
    accept_temp : np.ndarray
        Boolean array indicating which samples passed the RMSE constraint.
    samples[accept_temp] : np.ndarray
        Array of sample IDs that passed the RMSE constraint.

    Notes
    -----
    - The function assumes temperature data is aligned to time bounds, while observations are at midyears.
    - Baseline adjustment is performed using the average over the first 52 time steps, weighted by `weights`.
    - The function prints diagnostic information about shapes and the number of passing samples.
    """
    if file_endstring is None:
        file_endstring = ""
    gmst, weights = prepare_weights_temp(prune_list[1])
    samples = np.load(
        f"output/sample_ids_{total_samples}_chunk_{chunk_num}{file_endstring}.npy",
        allow_pickle=True,
    )
    temp_in = np.load(
        f"output/{prune_list[0]}_{total_samples}_chunk_{chunk_num}{file_endstring}_1850-2023.npy"
    )
    rmse_accept = prune_list[2]
    print(temp_in.shape)
    print(samples.shape)
    # temperature is on timebounds, and observations are midyears
    # but, this is OK, since we are subtracting a consistent baseline (1850-1900, weighting
    # the bounding timebounds as 0.5)
    # e.g. 1993.0 timebound has big pinatubo hit, timebound 143
    # in obs this is 1992.5, timepoint 142
    # compare the timebound after the obs, since the forcing has had chance to affect both
    # the obs timepoint and the later timebound.
    # the goal of RMSE is as much to match the shape of warming as the magnitude; we do not
    # want to average out internal variability in the model or the obs.
    rmse_temp = np.zeros(len(samples))
    for i in tqdm(range(len(samples))):
        rmse_temp[i] = rmse(
            gmst[:173],
            temp_in[i, 1:] - np.average(temp_in[i, :52], weights=weights, axis=0),
        )
    accept_temp = rmse_temp < rmse_accept
    print("Passing RMSE constraint:", np.sum(accept_temp))
    valid_temp = np.arange(len(samples), dtype=int)[accept_temp]
    return valid_temp, accept_temp, samples[accept_temp]


def get_targ_paramat_valid_for_chunk(
    chunk_num, valid_samples, total_samples=6000000, file_endstring=""
):
    """
    Retrieve target and parameter matrices for specified valid sample indices from an HDF5 chunk file.

    Parameters
    ----------
    chunk_num : int
        The chunk number identifying the HDF5 file to load.
    valid_samples : array-like
        Indices of valid samples to select from the target and parameter matrices.
    total_samples : int, optional
        The total number of samples used in the filename pattern (default is 6,000,000).
    file_endstring : str, optional
        Optional suffix to append to filenames (default is "").

    Returns
    -------
    targ : pandas.DataFrame
        DataFrame containing the selected rows from the target matrix.
    parammat : pandas.DataFrame
        DataFrame containing the selected rows from the parameter matrix.
    """
    store = pd.HDFStore(
        f"output/data_{total_samples}_chunk_{chunk_num}{file_endstring}.h5"
    )
    targ = store["targ"]
    parammat = store["parammat"]
    store.close()
    # print(targ.shape)
    # print(targ.head())
    # print(parammat.shape)
    # print(parammat.head())

    return targ.iloc[valid_samples, :], parammat.iloc[valid_samples, :]


def prune_all_chunks(total_samples, prune_lists, num_chunks=600, file_endstring=None):
    """
    Prunes samples across multiple data chunks and aggregates valid indices, sample IDs, targets, and parameter matrices.

    Parameters
    ----------
    total_samples : int
        The total number of samples to consider for pruning.
    prune_lists : list of list
        A list containing pruning criteria lists for each variable or chunk.
    num_chunks : int, optional
        The number of data chunks to process (default is 600).
    file_endstring : str or None, optional
        String to append to output filenames for identification (default is None).

    Returns
    -------
    None
        This function saves the pruned results to disk as `.npy` and `.h5` files.

    Notes
    -----
    - Currently, pruning for multiple variables (i.e., when `len(prune_lists) > 1`) is not implemented.
    - The function relies on external functions `do_pruning_for_chunk` and `get_targ_paramat_valid_for_chunk`.
    - Output files are saved in the `data/` directory with names including `file_endstring` if provided.
    """
    keep_temp = []
    keep_samples = []
    keep_targ = []
    keep_parammat = []
    # TODO: expand to pruning for multiple variables
    if len(prune_lists) > 1:
        print("Currently unimplemented. TODO")

    for chunk_num in tqdm(range(num_chunks)):
        prune_list = prune_lists[0]
        valid_temp, accept_temp, samples_keep = do_pruning_for_chunk(
            chunk_num=chunk_num,
            prune_list=prune_list,
            file_endstring=file_endstring,
            total_samples=total_samples,
        )
        print(samples_keep)
        print(accept_temp)
        print(valid_temp)
        print(len(valid_temp))
        keep_temp.append(valid_temp)
        keep_samples.append(samples_keep)
        targ_keep, parammat_keep = get_targ_paramat_valid_for_chunk(
            chunk_num,
            valid_temp,
            total_samples=total_samples,
            file_endstring=file_endstring,
        )
        print(targ_keep.shape)
        print(parammat_keep.shape)
        keep_targ.append(targ_keep)
        keep_parammat.append(parammat_keep)

    valid_temps_all = np.concatenate(keep_temp)
    np.save(f"output/valid_indices_all_chunks{file_endstring}.npy", valid_temps_all)
    valid_ids_all = np.concatenate(keep_samples)
    np.save(f"output/valid_sample_ids_all_chunks{file_endstring}.npy", valid_ids_all)
    all_targs = pd.concat(keep_targ)
    all_paramat = pd.concat(keep_parammat)

    store = pd.HDFStore(f"output/data_all_targs_paramats{file_endstring}.h5")
    store["targ"] = all_targs
    store["parammat"] = all_paramat
    store.close()


if __name__ == "__main__":
    prune_all_chunks()
# sys.exit(4)

# CO2 pruning block
# co2_conc = plot_distributions_w_obs.read_noaa_gml_ml_means("year")
# co2_in = targ["Atmospheric Concentrations|CO2"].to_numpy()
# accepted_co2 = (np.abs(co2_in - 410) < 5)
# print("Passing CO2 constraint:", np.sum(accepted_co2))
# valid_co2 = np.arange(len(samples), dtype=int)[accepted_co2]


# plot_full_dist = False
# sys.exit(4)


# np.savetxt(
#    f"data/temp_{len(samples)}_runids_rmse_pass.csv",
#    valid_co2.astype(int),
#    fmt="%d",
# )

# valid_both = np.array(list(set(valid_co2).intersection(set(valid_temp))), dtype=int)
# print(f"Valid for both constraints: {len(valid_both)}")
# print(valid_both)

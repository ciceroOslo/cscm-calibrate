
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm.auto import tqdm

from .plot_distributions_w_obs  import datadir
from .shared_functions import rmse


def prepare_weights_temp(data_path, data_varname = "GMST"):
    weights = np.ones(52)
    weights[0] = 0.5
    weights[-1] = 0.
    # Temperature pruning input

    temp_data = pd.read_csv(data_path)
    gmst = temp_data[data_varname].values
    return gmst, weights

def do_pruning_for_chunk(chunk_num, prune_list, file_endstring = None, total_samples= 6000000):
    if file_endstring is None:
        file_endstring = ""
    gmst, weights = prepare_weights_temp(prune_list[1])
    samples = np.load(f"data/sample_ids_{total_samples}_chunk_{chunk_num}{file_endstring}.npy", allow_pickle=True)
    temp_in = np.load(f"data/{prune_list[0]}_{total_samples}_chunk_{chunk_num}_1850-2023.npy")
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
            temp_in[i, 1:] - np.average(temp_in[i,:52], weights=weights, axis=0),
        )
    accept_temp = rmse_temp < rmse_accept
    print("Passing RMSE constraint:", np.sum(accept_temp))
    valid_temp = np.arange(len(samples), dtype=int)[accept_temp]
    return valid_temp, accept_temp, samples[accept_temp]

def get_targ_paramat_valid_for_chunk(chunk_num, valid_samples, total_samples= 6000000):
    store = pd.HDFStore(f'data/data_{total_samples}_chunk_{chunk_num}.h5')
    targ = store['targ']
    parammat = store['parammat']
    store.close()
    #print(targ.shape)
    #print(targ.head())
    #print(parammat.shape)
    #print(parammat.head())

    return targ.iloc[valid_samples, :], parammat.iloc[valid_samples, :]

def prune_all_chunks(total_samples, prune_lists, num_chunks=600, file_endstring = None):

    keep_temp = []
    keep_samples = []
    keep_targ = []
    keep_parammat = []
    # TODO: expand to pruning for multiple variables
    if len(prune_lists) > 1:
        print("Currently unimplemented. TODO")

    for chunk_num in tqdm(range(num_chunks)):
        prune_list = prune_lists[0]
        valid_temp, accept_temp, samples_keep = do_pruning_for_chunk(chunk_num=chunk_num, prune_list=prune_list, file_endstring=file_endstring, total_samples=total_samples)
        print(samples_keep)
        print(accept_temp)
        print(valid_temp)
        print(len(valid_temp))
        keep_temp.append(valid_temp)
        keep_samples.append(samples_keep)
        targ_keep, parammat_keep = get_targ_paramat_valid_for_chunk(chunk_num, valid_temp, total_samples= 6000000)
        print(targ_keep.shape)
        print(parammat_keep.shape)
        keep_targ.append(targ_keep)
        keep_parammat.append(parammat_keep)

    valid_temps_all = np.concatenate(keep_temp)
    np.save(f"data/valid_indices_all_chunks{file_endstring}.npy", valid_temps_all)
    valid_ids_all = np.concatenate(keep_samples)
    np.save(f"data/valid_sample_ids_all_chunks{file_endstring}.npy", valid_ids_all)
    all_targs = pd.concat(keep_targ)
    all_paramat = pd.concat(keep_parammat)

    store = pd.HDFStore(f"data/data_all_targs_paramats{file_endstring}.h5")
    store['targ'] = all_targs
    store['parammat'] = all_paramat
    store.close()


if __name__ == "__main__":
    prune_all_chunks()
#sys.exit(4)

# CO2 pruning block
#co2_conc = plot_distributions_w_obs.read_noaa_gml_ml_means("year")
#co2_in = targ["Atmospheric Concentrations|CO2"].to_numpy()
#accepted_co2 = (np.abs(co2_in - 410) < 5)
#print("Passing CO2 constraint:", np.sum(accepted_co2))
#valid_co2 = np.arange(len(samples), dtype=int)[accepted_co2]



#plot_full_dist = False
#sys.exit(4)






#np.savetxt(
#    f"data/temp_{len(samples)}_runids_rmse_pass.csv",
#    valid_co2.astype(int),
#    fmt="%d",
#)

#valid_both = np.array(list(set(valid_co2).intersection(set(valid_temp))), dtype=int)
#print(f"Valid for both constraints: {len(valid_both)}")
#print(valid_both)

    



import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import plot_distributions_w_obs 


# In[15]:

def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))


do_pam_plotting = False

if do_pam_plotting: 
    store = pd.HDFStore('data/data.h5')
    targ = store['targ']
    parammat = store['parammat']

    store.close()
    print(targ)
    print(parammat)
    plot_distributions_w_obs.pam_plotting(paramat=parammat)

weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

temp_data = pd.read_csv(f"{plot_distributions_w_obs.datadir}annual_averages.csv")
gmst = temp_data["GMST"].values
print(gmst)
print(temp_data.head)
print(len(gmst))
co2_conc = plot_distributions_w_obs.read_noaa_gml_ml_means("year")

store_long = pd.HDFStore("data/data_long.h5")
results_full = store_long["results"]
store_long.close()
print(results_full.shape)
print(results_full.head())
samples = results_full.index.values
temp_in = results_full.loc[results_full["variable"] =="Surface Air Ocean Blended Temperature Change"].iloc[:,7:].to_numpy(float)
samples = results_full.loc[results_full["variable"] =="Surface Air Ocean Blended Temperature Change"].iloc[:,2].to_numpy(int)
print(temp_in.shape)
print(samples.shape)
plot_full_dist = True
if plot_full_dist:
    plot_distributions_w_obs.plot_distributions(results_full, f"{len(samples)}_first_test_ts")
#sys.exit(4)

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
        temp_in[i, 101:] - np.average(temp_in[i,100:152], weights=weights, axis=0),
    )

accept_temp = rmse_temp < 0.17
print("Passing RMSE constraint:", np.sum(accept_temp))
valid_temp = np.arange(len(samples), dtype=int)[accept_temp]

# get 10 largest (but passing) and 10 smallest RMSEs
rmse_temp_accept = rmse_temp[accept_temp]
just_passing = np.argpartition(rmse_temp_accept, -10)[-10:]
smashing_it = np.argpartition(rmse_temp_accept, 10)[:10]
print(just_passing)
print(rmse_temp_accept[just_passing])
print(rmse_temp_accept[smashing_it])
#valid_temp = np.arange(samples, dtype=int)[accept_temp]
np.savetxt(
    f"data/{len(samples)}_runids_rmse_pass.csv",
    valid_temp.astype(int),
    fmt="%d",
)
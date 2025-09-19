
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plot_distributions_w_obs 


# In[15]:

do_pam_plotting = False

if do_pam_plotting: 
    store = pd.HDFStore('data/data.h5')
    targ = store['targ']
    parammat = store['parammat']

    store.close()
    print(targ)
    print(parammat)
    plot_distributions_w_obs.pam_plotting(paramat=parammat)

store_long = pd.HDFStore("data/data_long.h5")
results_full = store_long["results"]
store_long.close()
print(results_full.shape)
print(results_full.head())
plot_distributions_w_obs.plot_distributions(results_full, "50000_first_test_ts.png")

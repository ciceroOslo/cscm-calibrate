#!/usr/bin/env python
# coding: utf-8

# # CICERO SCM notebook - parallel application

# Import some stuff

# In[1]:


import sys
import re
import os
import numpy as np
import pandas as pd
import pandas.testing as pdt
import warnings

import plot_distributions_w_obs

try:
    from pandas.core.common import SettingWithCopyWarning
except:
    from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", message=".*Parameter.*")

cscm_path = os.path.join("..", "..", "..", "ciceroscm")

sys.path.insert(0,os.path.join(cscm_path, 'src'))

from ciceroscm.parallel._configdistro import _ConfigDistro
from ciceroscm.parallel.distributionrun import DistributionRun

from ciceroscm import CICEROSCM

from ciceroscm import input_handler


test_data_dir = os.path.join(os.getcwd(), '../../', 'data', 'calibration_data_Sep2025')
gaspam = input_handler.read_components(test_data_dir + '/gases_vupdate_2024_WMO_added_new.txt')
print(gaspam.head())
nyend = 2023

df_nat_ch4 = input_handler.read_natural_emissions(test_data_dir + '/natemis_CH4_ode_method_from_Sep2025_updates.txt','CH4', endyear=nyend)
df_nat_n2o = input_handler.read_natural_emissions(test_data_dir + '/natemis_N2O_ode_method_from_Sep2025_updates.txt','N2O', endyear=nyend)
print(df_nat_ch4.head())


df_ssp2_conc =input_handler.read_inputfile(test_data_dir + '/igcc_historical_conc_gases_vupdate_2024_WMO_added_new.txt', True, year_end=nyend)

ih = input_handler.InputHandler({"nyend": nyend, "nystart": 1750, "emstart": 1850})
emi_input =ih.read_emissions(test_data_dir + '/historical_em_gases_vupdate_2024_WMO_added_new.txt')
emi_input.rename(columns={"CO2": "CO2_FF", "CO2.1": "CO2_AFOLU"}, inplace=True)


scendata={
            "gaspam_data": gaspam,
            "emstart": 1850,  
            "conc_run":False,
            "nystart": 1750,
            "nyend": nyend,
            "concentrations_data": df_ssp2_conc,
            "emissions_data": emi_input,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm":24,
            "udir": test_data_dir,
            "scenname": "ssp245",
        }

prior_distro_dict = {
    "rlamdo": [5, 25],
    "akapa": [0.06, 0.8],
    "cpi": [0.161, 0.569],
    "W": [0.55, 2.55],
    "beto": [0, 7],
    "lambda": [2 / 3.71, 5 / 3.71],
    "mixed": [25, 125],
    "qo3": [0.4, 0.6],
    "qdirso2": [-0.005, -0.000],
    "qindso2": [-0.02, -0.00],
    "qbc": [0.004, 0.05],
    "qoc": [-0.008, -0.001],
    "beta_f": [0.110, 1.0],
    "mixed_carbon": [25, 125],
    "solubility_sens": [0, 0.03],
    "ocean_efficacy": [0.9, 1.3],
    "ml_w_sigmoid": [2.0, 7.0],
    "ml_fracmax": [0., 1.0],
    "t_half": [0.3, 0.8],
    "t_threshold": [3, 10],
    "w_threshold": [2,8],
    "w_sigmoid": [2,8]
}

prior_distro_dict_just_carbon = {
    "beta_f": [0.110, 0.465],
    "mixed_carbon": [25, 125],
    "solubility_sens": [0, 0.03],
    "ml_w_sigmoid": [2.0, 7.0],
    "ml_fracmax": [0., 1.0],
    "npp0": [50, 70],
    "t_half": [0.3, 0.8],
    "t_threshold": [3, 10],
    "w_threshold": [2,8],
    "w_sigmoid": [2,8]
}


testconfig = _ConfigDistro(
    distro_dict=prior_distro_dict,
    setvalues={
        "threstemp": 7.0,
        "lm": 40,
        "ldtime": 12,
        "qbmb": 0,
        "solubility_limit": 0.1,
        "ml_t_half": 0.
    },
)

scen = 'test'
cscm_dir=CICEROSCM({
            "gaspam_data": gaspam,
            "emstart": 1850,  
            "conc_run":False,
            "nystart": 1750,
            "nyend": nyend,
            "concentrations_data": df_ssp2_conc,
            "emissions_data": emi_input,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm":24
        })

scenariodata = [{
            "gaspam_data": gaspam,
            "emstart": 1850,  
            "conc_run":False,
            "nystart": 1750,
            "nyend": nyend,
            "concentrations_data": df_ssp2_conc,
            "emissions_data": emi_input,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm":24,
            "scenname" : "ssp245-short"
        }]

calibdata = pd.DataFrame(
    data={
        "Variable Name": [
            "Heat Content|Ocean",
            "Surface Air Ocean Blended Temperature Change",
            "Effective Radiative Forcing|Aerosols",
            "Atmospheric Concentrations|CO2",
            "Ocean carbon flux", 
            "Biosphere carbon flux" 
        ],
        "Yearstart_norm": [1971, 1850, 1750, 1750, 1750, 1750],
        "Yearend_norm": [1971, 1900, 1750, 1750, 1750, 1750],
        "Yearstart_change": [2023, 2011, 2023, 2023, 2014, 2014],
        "Yearend_change": [2023, 2020, 2023, 2023, 2023, 2023],
        "Central Value": [484.82157000000063, 1.24, -1.18, 410.1-278, 2.9, 3.2],
        "sigma": [36.8551891022091 , 0.073, 0.7, 0.4, 0.4, 0.9],

    }
    )
# Recheck / rethink asymmetric uncertainty intervals, in particular for aerosol forcing
calibdata_longer_input = pd.DataFrame(
    data={
        "Variable Name": [
            "Heat Content|Ocean",
            "Surface Air Ocean Blended Temperature Change",
            "Effective Radiative Forcing|Aerosols",
            "Atmospheric Concentrations|CO2",
            "Ocean carbon flux", 
            "Biosphere carbon flux" 
        ],
        "Yearstart_norm": [1971, 1850, 1750, 1750, 1750, 1750],
        "Yearend_norm": [1971, 1900, 1750, 1750, 1750, 1750],
        "Yearstart_change": [2023, 2015, 2024, 2024, 2023, 2023],
        "Yearend_change": [2023, 2024, 2024, 2024, 2023, 2023],
        "Central Value": [484.82157000000063, 1.24, -1.07, 422.5, 2.880720693, 2.302042382],
        "sigma": [36.8551891022091 , 0.073, 0.7, 3.0, 0.4, 0.5],
    }
)
distnums = 1000000
distrorun1 = DistributionRun(testconfig, numvalues= distnums)
output_vars = calibdata["Variable Name"]
results = distrorun1.run_over_distribution(scenariodata, output_vars, max_workers=200)

print(results.head())
print(results.shape)

results_for_fit_dict_1d = {}
results_for_pruning = {}
for idx, data in calibdata.iterrows():
    print(data)
    results_sub = results.loc[results["variable"] == data["Variable Name"]]
    if data["Yearstart_norm"] == data["Yearend_norm"] and data["Yearstart_norm"]== 1750:
        results_for_fit_dict_1d[data["Variable Name"]] = results_sub.iloc[:, data["Yearstart_change"]-1750+7].values
    elif data["Yearstart_norm"] == data["Yearend_norm"]:
        results_for_fit_dict_1d[data["Variable Name"]] = (results_sub.iloc[:, data["Yearstart_change"]-1750+7] - results_sub.iloc[:,data["Yearstart_norm"]-1750+7]).values
    else:
        results_for_fit_dict_1d[data["Variable Name"]] = ((
            results_sub.iloc[:,data["Yearstart_change"]-1750+7: data["Yearend_change"]-1750+8]).mean(axis = 1) - (
            results_sub.iloc[:,data["Yearstart_norm"]-1750+7: data["Yearend_norm"]-1750+8]).mean(axis = 1)
        ).values
print(results_for_fit_dict_1d)

"""
print(results.loc[results["variable"] =="Atmospheric Concentrations|CO2"].values[:, 7:])
temp_targ = pd.DataFrame(
    data = results.loc[results["variable"]=="Surface Air Ocean Blended Temperature Change"].values[:, 7:],
    index=results.loc[results["variable"]=="Surface Air Ocean Blended Temperature Change"]["run_id"]
)

co2_targ = pd.DataFrame(
    data = results.loc[results["variable"] =="Atmospheric Concentrations|CO2"].values[:, 7:],
    index= results.loc[results["variable"] =="Atmospheric Concentrations|CO2"]["run_id"]
)
print(temp_targ)
print(co2_targ)
"""
plot = True
if plot:
    plot_distributions_w_obs.plot_distributions(results, f"{distnums}_test1_full.png")
    print(distrorun1.cfgs)
# There used to be a pruning sanity check here...
store = True
if store:
    targ = pd.DataFrame(
    data = results_for_fit_dict_1d
    )
    targ.index.set_names("run_id", inplace=True)
    pdict=distrorun1.cfgs

    def merge_dicts(dc):
        x=dc['pamset_udm']
        y=dc['pamset_emiconc']
        w=dc['pamset_carbon']
        z = x.copy()
        z.update(y)
        z.update(w)
        return z

    mdict=[ merge_dicts(d) for d in pdict ]
    pmat=pd.DataFrame(mdict)


    parammat=pmat.loc[:, (pmat != pmat.iloc[0]).any()]
    parammat

    #sys.exit(4)
    store = pd.HDFStore('data/data.h5')
    store['targ'] = targ
    store['parammat'] = parammat
    store.close()

    store_long = pd.HDFStore('data/data_long.h5')
    store_long['results'] = results
    store_long.close()


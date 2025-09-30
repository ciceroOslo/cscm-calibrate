
import sys
import os
import json
import pandas as pd
import numpy as np


cscm_path = os.path.join("..", "..", "..", "ciceroscm")

sys.path.insert(0,os.path.join(cscm_path, 'src'))

from ciceroscm.parallel._configdistro import ordering_standard_forc
from ciceroscm.carbon_cycle.carbon_cycle_mod import CARBON_CYCLE_MODEL_REQUIRED_PAMSET

def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))

def make_config_distro_json(matrix, parameter_names, json_name, indexer_pre="", index_list =None):
    config_list = [None] * matrix.shape[1]
    
    if index_list is None:
        index_list = [f"{indexer_pre}{i}" for i in matrix.shape[1]]

    for i in range(len(config_list)):
        pamset_udm = {
            "threstemp": 7.0,
            "lm": 40,
            "ldtime": 12,
            }
        pamset_emiconc = {"qbmb": 0,}
        pamset_carbon = {        
            "solubility_limit": 0.1,
            "ml_t_half": 0.
        }
        for j, pam in enumerate(parameter_names):
            value = matrix[j, i]
            if pam in ordering_standard_forc:
                pamset_udm[pam] = value
            elif pam in CARBON_CYCLE_MODEL_REQUIRED_PAMSET:
                pamset_carbon[pam] = value
            else:
                pamset_emiconc[pam] = value
        config_list[i] = {
                "pamset_udm": pamset_udm.copy(),
                "pamset_emiconc": pamset_emiconc.copy(),
                "pamset_carbon": pamset_carbon.copy(),
                "Index": index_list,
        }
    with open(f"data/{json_name}", "w", encoding="utf-8") as wfile:
        json.dump(config_list, wfile)
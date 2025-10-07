import sys
import os
import json
import pandas as pd
import numpy as np


cscm_path = cscm_path = "/home/masan/gitrepos/ciceroscm"#os.path.join("..", "..", "..", "ciceroscm")


sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.parallel._configdistro import ordering_standard_forc
from ciceroscm.carbon_cycle.carbon_cycle_mod import CARBON_CYCLE_MODEL_REQUIRED_PAMSET


def rmse(obs, mod):
    """
    Calculate the Root Mean Square Error (RMSE) between observed and modeled values.
    Parameters
    ----------
    obs : array-like
        Array of observed values.
    mod : array-like
        Array of modeled or predicted values.
    Returns
    -------
    float
        The root mean square error between the observed and modeled values.
    Notes
    -----
    Both `obs` and `mod` must be of the same length.
    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> mod = np.array([1.1, 1.9, 3.2])
    >>> round(rmse(obs, mod), 8)
    np.float64(0.14142136)
    """
    return np.sqrt(np.mean((obs - mod) ** 2))


def make_config_distro_json(
    matrix, parameter_names, json_name, indexer_pre="", index_list=None
):
    """
    Generates a JSON file containing a list of configuration dictionaries based on the provided parameter matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        A 2D numpy array where each column represents a set of parameter values for a configuration.
    parameter_names : list of str
        List of parameter names corresponding to the rows of the matrix.
    json_name : str
        The name of the output JSON file (will be saved in the 'data/' directory).
    indexer_pre : str, optional
        Prefix to use for generating index names if `index_list` is not provided (default is an empty string).
    index_list : list of str, optional
        List of index names for each configuration. If None, index names are generated using `indexer_pre`.
    
    Returns
    -------
    None
        The function writes the configuration list to a JSON file and does not return anything.
   
    Notes
    -----
    The function expects the global variables `ordering_standard_forc` and `CARBON_CYCLE_MODEL_REQUIRED_PAMSET` 
    to be defined elsewhere in the code. Each configuration dictionary contains three parameter sets 
    ('pamset_udm', 'pamset_emiconc', 'pamset_carbon') and an 'Index' field.
    """
    config_list = [None] * matrix.shape[1]

    if index_list is None:
        index_list = [f"{indexer_pre}{i}" for i in range(matrix.shape[1])]

    for i in range(len(config_list)):
        pamset_udm = {
            "threstemp": 7.0,
            "lm": 40,
            "ldtime": 12,
        }
        pamset_emiconc = {
            "qbmb": 0,
        }
        pamset_carbon = {"solubility_limit": 0.1, "ml_t_half": 0.0}
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

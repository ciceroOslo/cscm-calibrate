import json
import os
import sys

import pandas as pd
import numpy as np

# Get path to ciceroscm - one level up from project root
cscm_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ciceroscm")
)

sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.carbon_cycle.carbon_cycle_mod import CARBON_CYCLE_MODEL_REQUIRED_PAMSET
from ciceroscm.parallel._configdistro import ordering_standard_forc

SIGMA_TO_90PERCENT = 1.6448536269514722

varname_short_mapping = {
    "Heat Content|Ocean": "OHC",
    "Surface Air Ocean Blended Temperature Change": "GMST",
    "Effective Radiative Forcing|Aerosols": "ERFaer",
    "Atmospheric Concentrations|CO2": "CO2conc",
    "Ocean carbon flux": "Oceancarbon",
    "Biosphere carbon flux": "Biocarbon",
}

RCMIP_NAME_MAPPING = {
    "Global Mean Surface Temperature (GMST)": "Surface Air Ocean Blended Temperature Change",
    "Ocean Heat Content|Global|Total": "Heat Content|Ocean",
    "Carbon Flux to Oceans": "Ocean carbon flux",
    "Carbon Flux to Land": "Biosphere carbon flux",
}

def get_project_root():
    """Get the project root directory (where this package is installed)."""
    # Go up from src/cscm_calibrate/ to the project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


def make_constraints_config_from_RCMIP_csv(constraints_from_RCMIP):
    """
    Reads constraint data from a CSV file and writes it to a JSON file.

    Parameters
    ----------
    constraints_from_RCMIP : str
        Path to the input CSV file containing constraint data.
    output_json_name : str
        Name of the output JSON file to write the constraints to.

    Returns
    -------
    None
        The function writes the constraints to a JSON file and does not return anything.
    """

    constraints_df = pd.read_csv(constraints_from_RCMIP)
    constraints_dict = {
        "Variable Name": [],
        "Varname_short": [],
        "Yearstart_norm": [],
        "Yearend_norm": [],
        "Yearstart_change": [],
        "Yearend_change": [],
        "Central Value": [],
        "lower_sigma": [],
        "upper_sigma": [],
        "run_experiments": [],
    }
    print(constraints_df)
    for rownum, row in constraints_df.iterrows():
        varname = row["Variable"]
        if varname in RCMIP_NAME_MAPPING:
            varname = RCMIP_NAME_MAPPING[varname]
        constraints_dict["Variable Name"].append(varname)
        constraints_dict["Varname_short"].append(varname_short_mapping[varname])
        constraints_dict["run_experiments"].append("historical")
        base_years = row["Baseline_period"].split("-")
        const_years = row["Constraint_period"].split("-")
        try:
            constraints_dict["Yearstart_norm"].append(int(base_years[0]))
            constraints_dict["Yearend_norm"].append(int(base_years[1]))
        except ValueError:
            constraints_dict["Yearstart_norm"].append(1750)
            constraints_dict["Yearend_norm"].append(1750)
        constraints_dict["Yearstart_change"].append(int(const_years[0]))
        constraints_dict["Yearend_change"].append(int(const_years[1]))
        central = float(row["Central_estimate"])
        constraints_dict["Central Value"].append(central)
        constraints_dict["lower_sigma"].append(
            (central - float(row["Lower_bound"])) / SIGMA_TO_90PERCENT
        )
        constraints_dict["upper_sigma"].append(
            (float(row["Upper_bound"]) - central) / SIGMA_TO_90PERCENT
        )

    print(constraints_dict)
    # Convert to DataFrame for compatibility with run_prior_ensemble
    return pd.DataFrame(constraints_dict)

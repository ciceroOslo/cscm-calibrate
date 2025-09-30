#!/usr/bin/env python
# coding: utf-8

"""Apply posterior weighting"""

# mention in paper: skew-normal distribution
# this is where Zeb earns his corn

import os, sys

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

import plot_distributions_w_obs

#from prune_distribution_to_timeseries import make_config_distro_json

cscm_path = os.path.join("..", "..", "..", "ciceroscm")

sys.path.insert(0,os.path.join(cscm_path, 'src'))

from ciceroscm.parallel._configdistro import ordering_standard_forc
from ciceroscm.carbon_cycle.carbon_cycle_mod import CARBON_CYCLE_MODEL_REQUIRED_PAMSET


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
                "Index": index_list[i],
        }
    with open(f"data/{json_name}", "w", encoding="utf-8") as wfile:
        json.dump(config_list, wfile)

plt.switch_backend("agg")

print("Doing reweighting...")

store = pd.HDFStore('data/data_all_targs_paramats.h5')
targ = store['targ']
parammat = store['parammat']

store.close()
print(targ)
print(parammat)
print(targ.shape)
print(parammat.shape)
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

# Only needed if not chunked
not_chunked = False
if not_chunked:
    valid_temp = np.loadtxt(
        f"data/{targ.shape[0]}_runids_rmse_pass.csv"
    ).astype(np.int64)

    #print(valid_temp)

output_ensemble_size = 500
input_ensemble_size = targ.shape[0]

#sys.exit(4)

assert input_ensemble_size > output_ensemble_size

temp_in = targ["Surface Air Ocean Blended Temperature Change"].to_numpy()#[valid_temp]
ohc_in = targ["Heat Content|Ocean"].to_numpy()#[valid_temp]
faer_in = targ["Effective Radiative Forcing|Aerosols"].to_numpy()#[valid_temp]
co2_in = targ["Atmospheric Concentrations|CO2"].to_numpy()#[valid_temp]
occarb_in = targ["Ocean carbon flux"].to_numpy()#[valid_temp]
biocarb_in = targ["Ocean carbon flux"].to_numpy()#[valid_temp]

print(len(temp_in))

def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)

samples = {}

gsat_params = scipy.optimize.root(opt, [1, 1, 1], args=(0.95, 1.09, 1.20)).x
erf_params = scipy.optimize.root(opt, [1, 1, 1], args=(-2.10, -1.18, -0.49)).x
print(erf_params)
print(gsat_params)
# note fair produces, and we here report, total earth energy uptake, not just ocean
# this value from IGCC 2023. Use new uncertainties for ocean, assume same uncertainties
# for land, atmosphere and cryopshere.
samples["OHC"] = scipy.stats.norm.rvs(
    loc=484.8, scale=36.9, size=10**5, random_state=43178
)
samples["temperature 2011-2020"] = scipy.stats.skewnorm.rvs(
    gsat_params[0],
    loc=gsat_params[1],
    scale=gsat_params[2],
    size=10**5,
    random_state=19387,
)
samples["ERFaer"] = scipy.stats.skewnorm.rvs(
    erf_params[0],
    loc=erf_params[1],
    scale=erf_params[2],
    size=10**5,
    random_state=91123,
)
# IGCC paper: 417.1 +/- 0.4
# IGCC dataset: 416.9
# my assessment 417.0 +/- 0.5
samples["CO2 concentration"] = scipy.stats.norm.rvs(
    loc=419.3, scale=0.5, size=10**5, random_state=81693
)

samples["Ocean carbon 2014-2023"] = scipy.stats.norm.rvs(
    loc=2.9, scale=0.4, size=10**5, random_state=81693
)
samples["Biosphere carbon 2014-2023"] = scipy.stats.norm.rvs(
    loc=3.2, scale=0.9, size=10**5, random_state=81693
)

ar_distributions = {}
for constraint in [
    "OHC",
    "temperature 2011-2020",
    "ERFaer",
    "CO2 concentration",
    "Ocean carbon 2014-2023",
    "Biosphere carbon 2014-2023"
]:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
        samples[constraint], bins=100, density=True
    )[1]
    ar_distributions[constraint]["values"] = samples[constraint]
"""
weights_20yr = np.ones(21)
weights_20yr[0] = 0.5
weights_20yr[-1] = 0.5
weights_51yr = np.ones(52)
weights_51yr[0] = 0.5
weights_51yr[-1] = 0.5

co2_1850 = 284.3169988
co2_1920 = co2_1850 * 1.01**70  # NOT 2x (69.66 yr), per definition of TCRE
"""
if not_chunked:
    accepted = pd.DataFrame(
        {
            "OHC": ohc_in[valid_temp],
            "temperature 2011-2020": temp_in[valid_temp],
            "ERFaer": faer_in[valid_temp],
            "CO2 concentration": co2_in[valid_temp],
            "Ocean carbon 2014-2023": occarb_in[valid_temp],
            "Biosphere carbon 2014-2023": biocarb_in[valid_temp]
        },
        index=valid_temp,
    )
else:
    accepted = pd.DataFrame(
        {
            "OHC": ohc_in,
            "temperature 2011-2020": temp_in,
            "ERFaer": faer_in,
            "CO2 concentration": co2_in,
            "Ocean carbon 2014-2023": occarb_in,
            "Biosphere carbon 2014-2023": biocarb_in
        },
        index=np.arange(targ.shape[0]),
    )
def calculate_sample_weights(distributions, samples, niterations=50):
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in tqdm(
        range(niterations), desc="Iterations", leave=False#, disable=1 - progress
    ):
        gofs.append([])
        if k == (niterations - 1):
            weights_second_last_iteration = weights.copy()
            weights_to_average = []

        for j, unique_code in enumerate(unique_codes):
            unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                unique_code, distributions, samples, weights, j, k
            )
            if k == (niterations - 1):
                weights_to_average.append(unique_code_weights[our_values_bin_idx])

            weights *= unique_code_weights[our_values_bin_idx]

            gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
            gofs[-1].append(gof)

            gofs_full.append([unique_code])
            for unique_code_check in unique_codes:
                unique_code_check_weights, _ = get_unique_code_weights(
                    unique_code_check, distributions, samples, weights, 1, 1
                )
                gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                gofs_full[-1].append(gof)

    weights_stacked = np.vstack(weights_to_average).mean(axis=0)
    weights_final = weights_stacked * weights_second_last_iteration

    gofs_full.append(["Final iteration"])
    for unique_code_check in unique_codes:
        unique_code_check_weights, _ = get_unique_code_weights(
            unique_code_check, distributions, samples, weights_final, 1, 1
        )
        gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
        gofs_full[-1].append(gof)

    return (
        weights_final,
        pd.DataFrame(np.array(gofs), columns=unique_codes),
        pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
    )


def get_unique_code_weights(unique_code, distributions, samples, weights, j, k):
    bin_edges = distributions[unique_code]["bins"]
    our_values = samples[unique_code].copy()

    our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
    np.testing.assert_allclose(bin_edges, bin_edges_np)
    assessed_ranges_bin_counts, _ = np.histogram(
        distributions[unique_code]["values"], bins=bin_edges
    )

    our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

    existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
    for i in range(existing_weighted_bin_counts.shape[0]):
        existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

    if np.equal(j, 0) and np.equal(k, 0):
        np.testing.assert_equal(
            existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
        )

    unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

    # existing_weighted_bin_counts[0] refers to samples outside the
    # assessed range's lower bound. Accordingly, if `our_values` was
    # digitized into a bin idx of zero, it should get a weight of zero.
    unique_code_weights[0] = 0
    # Similarly, if `our_values` was digitized into a bin idx greater
    # than the number of bins then it was outside the assessed range
    # so get a weight of zero.
    unique_code_weights[-1] = 0

    for i in range(1, our_values_bin_counts.shape[0] + 1):
        # the histogram idx is one less because digitize gives values in the
        # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
        histogram_idx = i - 1
        if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
            unique_code_weights[i] = 0
        elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
            # other variables force this box to be zero so just fill it with
            # one
            unique_code_weights[i] = 1
        else:
            unique_code_weights[i] = (
                assessed_ranges_bin_counts[histogram_idx]
                / existing_weighted_bin_counts[histogram_idx]
            )

    return unique_code_weights, our_values_bin_idx


weights, gofs, gofs_full = calculate_sample_weights(
    ar_distributions, accepted, niterations=60
)

plot_pam_distributions = True
if plot_pam_distributions:
    print(weights.shape)
    print(accepted.shape)
    #store = pd.HDFStore('data/data.h5')
    #targ = store['targ']
    #parammat = store['parammat']

    #store.close()
    print(targ)
    print(parammat)
    plot_distributions_w_obs.pam_plotting(parammat, name_epithet="post1")
    plot_distributions_w_obs.pam_plotting(parammat, weights= np.minimum(weights, 1), name_epithet="post2")


effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
print("Number of effective samples:", effective_samples)

assert effective_samples >= output_ensemble_size

draws = []
drawn_samples = accepted.sample(
    n=output_ensemble_size, replace=False, weights=weights, random_state=10099
)
print(drawn_samples)
sample_ids = np.load("data/valid_sample_ids_all_chunks.npy", allow_pickle=True)[drawn_samples.index.to_list()]
make_config_distro_json(parammat.iloc[drawn_samples.index.to_list(), :].to_numpy().transpose(), parammat.columns, f"draw_samples_{output_ensemble_size}.json", index_list=sample_ids)
sys.exit(4)
draws.append((drawn_samples))
name_dict = {
    "OHC": [ohc_in, 100, 900],
    "temperature 2011-2020": [temp_in, 0.6, 1.6],
    "ERFaer": [faer_in, -3, 0.4],
    "CO2 concentration": [co2_in, 405, 421],
    "Ocean carbon 2014-2023": [occarb_in, 1., 6.],
    "Biosphere carbon 2014-2023": [biocarb_in, 1., 6.] 
}

target_temp = scipy.stats.gaussian_kde(samples["temperature 2011-2020"])
prior_temp = scipy.stats.gaussian_kde(temp_in)
post1_temp = scipy.stats.gaussian_kde(temp_in[valid_temp])
post2_temp = scipy.stats.gaussian_kde(draws[0]["temperature 2011-2020"])

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

plots = True
if plots:
    fig, ax = plt.subplots(3, 3, figsize=(18 / 2.54, 18 / 2.54))
    for i, vari in enumerate(sorted(name_dict.keys())):
        targ_dist = scipy.stats.gaussian_kde(samples[vari])
        print(type(name_dict[vari][0]))
        print(name_dict[vari][0].shape)
        print(samples[vari].shape)
        print(vari)
        print(name_dict[vari][0])
        prior_dist = scipy.stats.gaussian_kde(name_dict[vari][0].astype(float))
        post1_dist = scipy.stats.gaussian_kde(name_dict[vari][0][valid_temp].astype(float))
        post2_dist = scipy.stats.gaussian_kde(draws[0][vari].astype(float))
        axnow = ax[i//3, i%3]
        start = name_dict[vari][1]
        stop = name_dict[vari][2]
        axnow.plot(
            np.linspace(start, stop, 1000),
            targ_dist(np.linspace(start, stop, 1000)),
            color=colors["target"],
            label="Target",
            lw=2,
        )
        axnow.plot(
            np.linspace(start, stop, 1000),
            prior_dist(np.linspace(start, stop, 1000)),
            color=colors["prior"],
            label="Prior",
            lw=2,
        )
        axnow.plot(
        np.linspace(start, stop, 1000),
        post1_dist(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
        )
        axnow.plot(
            np.linspace(start, stop, 1000),
            post2_dist(np.linspace(start, stop, 1000)),
            color=colors["post2"],
            label="All constraints",
            lw=2,
        )

        axnow.set_xlim(start, stop)
        #axnow.set_ylim(0, 0.6)
        axnow.set_title(vari)
        axnow.set_yticklabels([])
        #axnow.set_xlabel("°C")
        axnow.set_ylabel("Probability density")

    legend_lines = [
        Line2D([0], [0], color=colors["prior"], lw=2),
        Line2D([0], [0], color=colors["post1"], lw=2),
        Line2D([0], [0], color=colors["post2"], lw=2),
        Line2D([0], [0], color=colors["target"], lw=2),
    ]
    legend_labels = ["Prior", "Temperature RMSE", "All constraints", "Target"]
    ax[2, 1].legend(legend_lines, legend_labels, frameon=False, loc="upper left")

    fig.tight_layout()
    plt.savefig(
        f"plots/constraints.png"
    )
    plt.savefig(
        f"plots/constraints.pdf"
    )
    plt.close()
        

# move these to the validation script
print("Constrained, reweighted parameters:")
print(
    "CO2 concentration 2023:", np.percentile(draws[0]["CO2 concentration"], (5, 50, 95))
)
print(
    "Temperature 2011-2020 rel. 1850-1900:",
    np.percentile(draws[0]["temperature 2011-2020"], (5, 50, 95)),
)
print(
    "Aerosol ERF 2023 rel. 1750:",
    np.percentile(draws[0]["ERFaer"], (5, 50, 95)),
)
print("OHC change 2023 rel. 1971:", np.percentile(draws[0]["OHC"], (5, 50, 95)))
print("Mean annual ocean sink 2014-2023:", np.percentile(draws[0]["Ocean carbon 2014-2023"], (5, 50, 95)))
print("Mean annual land sink 2014-2023:", np.percentile(draws[0]["Biosphere carbon 2014-2023"], (5, 50, 95)))

print("*likely range")


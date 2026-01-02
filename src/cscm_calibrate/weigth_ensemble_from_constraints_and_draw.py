#!/usr/bin/env python

"""Apply posterior weighting"""

# mention in paper: skew-normal distribution
# this is where Zeb earns his corn

import os

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from tqdm.auto import tqdm

from .plot_distributions_w_obs import pam_plotting
from .shared_functions import make_config_distro_json, get_project_root

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)


def opt(x, q05_desired, q50_desired, q95_desired):
    """
    Compute the differences between the quantiles of a skew-normal distribution and desired quantile values.

    Given the parameters of a skew-normal distribution (`a`, `loc`, `scale`), this function calculates the 5th, 50th, and 95th percentiles (quantiles) of the distribution and returns their differences from the corresponding desired quantile values.

    Parameters
    ----------
    x : array-like of float
        Parameters of the skew-normal distribution in the order (a, loc, scale):
            - a : float
                Skewness parameter.
            - loc : float
                Location parameter (mean).
            - scale : float
                Scale parameter (standard deviation).
    q05_desired : float
        Desired value for the 5th percentile (0.05 quantile).
    q50_desired : float
        Desired value for the 50th percentile (0.50 quantile, median).
    q95_desired : float
        Desired value for the 95th percentile (0.95 quantile).

    Returns
    -------
    tuple of float
        Differences between the computed and desired quantiles:
            - (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)

    Notes
    -----
    This function is typically used as an objective function for optimization routines that fit a skew-normal distribution to match specified quantiles.
    """
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


def add_entry_to_sample_distributions(samples, constraint_config, varnum):
    """
    Adds a new entry to the `samples` dictionary by generating random samples for a variable
    based on its constraints, using either a normal or skew-normal distribution.

    Parameters
    ----------
    samples : dict
        Dictionary to which the generated samples will be added. The key is the variable name,
        and the value is a NumPy array of samples.
    constraint_config : dict
        Dictionary containing constraint information for variables. Must include the keys:
        "Varname_short" (list of variable names), "lower_sigma" (float), "upper_sigma" (float),
        and "Central Value" (float).
    varnum : int
        Index of the variable in "Varname_short" to process.

    Returns
    -------
    dict
        The updated `samples` dictionary with the new variable's samples added.

    Notes
    -----
    - If `lower_sigma` equals `upper_sigma`, a normal distribution is used.
    - If `lower_sigma` and `upper_sigma` differ, a skew-normal distribution is fitted.
    - The function assumes the existence of an `opt` function for optimization and that
      `scipy.stats` and `scipy.optimize` are imported.
    """
    print(varnum)
    name = constraint_config["Varname_short"][varnum]
    lower_sigma = constraint_config["lower_sigma"][varnum]
    upper_sigma = constraint_config["upper_sigma"][varnum]
    central = constraint_config["Central Value"][varnum]
    if lower_sigma == upper_sigma:
        samples[name] = scipy.stats.norm.rvs(
            loc=central, scale=lower_sigma, size=10**5, random_state=43178
        )
        return samples
    var_params = scipy.optimize.root(
        opt, [1, 1, 1], args=(central - lower_sigma, central, central + upper_sigma)
    ).x
    samples[name] = scipy.stats.skewnorm.rvs(
        var_params[0],
        loc=var_params[1],
        scale=var_params[2],
        size=10**5,
        random_state=19387,
    )
    return samples


def calculate_sample_weights(distributions, samples, niterations=50):
    """
    Iteratively calculates sample weights to match given marginal distributions using a reweighting scheme.

    Parameters
    ----------
    distributions : dict
        A dictionary where keys are unique codes (e.g., variable names or identifiers) and values are the target marginal distributions for each code.
    samples : np.ndarray
        Array of samples to be reweighted. Each row corresponds to a sample, and columns correspond to features or variables.
    niterations : int, optional
        Number of iterations to perform for the reweighting process (default is 50).

    Returns
    -------
    weights_final : np.ndarray
        The final computed weights for each sample after all iterations.
    gofs : pandas.DataFrame
        DataFrame containing the goodness-of-fit (gof) values for each unique code at each iteration.
    gofs_full : pandas.DataFrame
        DataFrame containing the goodness-of-fit values for all unique codes at each iteration, including the final iteration.

    Notes
    -----
    This function relies on an external function `get_unique_code_weights` to compute weights for each unique code.
    The reweighting process is performed iteratively to better match the target marginal distributions.
    """
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in tqdm(
        range(niterations),
        desc="Iterations",
        leave=False,  # , disable=1 - progress
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
    """
    Calculate and return the weights for each bin of a unique code's sample distribution,
    based on assessed ranges and existing sample weights.

    Parameters
    ----------
    unique_code : hashable
        The key identifying the distribution and samples to process.
    distributions : dict
        Dictionary containing distribution information for each unique code.
        Each entry should have "bins" (array-like) and "values" (array-like).
    samples : dict
        Dictionary containing sample arrays for each unique code.
    weights : np.ndarray
        Array of weights corresponding to the samples for the given unique code.
    j : int
        Index or flag used for assertion checks (typically for debugging or validation).
    k : int
        Index or flag used for assertion checks (typically for debugging or validation).

    Returns
    -------
    unique_code_weights : np.ndarray
        Array of weights for each bin, including underflow and overflow bins.
    our_values_bin_idx : np.ndarray
        Array of bin indices for each sample in `samples[unique_code]`, as returned by `np.digitize`.

    Notes
    -----
    - The first and last elements of `unique_code_weights` correspond to underflow and overflow bins, respectively.
    - If a bin in the assessed range has no samples, its weight is set to zero.
    - If a bin has no existing weighted samples, its weight is set to one.
    - The function asserts that the sum of weighted bin counts matches the sum of sample bin counts when `j == 0` and `k == 0`.
    """
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


def weight_ensemble_and_draw(
    constraint_config,
    file_endstring,
    output_ensemble_size=500,
    plot_pam_distributions=True,
):
    """
    Reweights an ensemble of samples based on constraints, draws a new ensemble, and optionally plots parameter distributions.

    This function loads target and parameter matrices from HDF5 files, applies constraints to reweight the ensemble,
    and draws a specified number of samples according to the computed weights. It can also plot the parameter distributions
    before and after reweighting. The selected samples and their configuration are saved to a JSON file.

    Parameters
    ----------
    constraint_config : dict
        Configuration dictionary specifying constraints and variable names for reweighting.
    file_endstring : str
        Suffix string used to locate input and output files.
    output_ensemble_size : int, optional
        Number of samples to draw for the output ensemble (default is 500).
    plot_pam_distributions : bool, optional
        If True, plots parameter distributions before and after reweighting (default is True).

    Returns
    -------
    None
        The function saves the drawn samples and their configuration to a JSON file and prints information to stdout.

    Raises
    ------
    AssertionError
        If the input ensemble size is not greater than the output ensemble size, or if the number of effective samples is less than the output ensemble size.
    """
    print("Doing reweighting...")

    output_dir = os.path.join(get_project_root(), "output")

    store = pd.HDFStore(
        os.path.join(output_dir, f"data_all_targs_paramats{file_endstring}.h5")
    )
    targ = store["targ"]
    parammat = store["parammat"]

    store.close()
    input_ensemble_size = targ.shape[0]

    # sys.exit(4)

    assert input_ensemble_size > output_ensemble_size
    data_in_dict = {}
    samples = {}
    ar_distributions = {}
    for varnum, constraint in enumerate(constraint_config["Varname_short"]):
        print(targ)
        print(constraint)
        print(constraint_config["Variable Name"][varnum])
        data_in_dict[constraint] = targ[
            constraint_config["Variable Name"][varnum]  
        ].to_numpy()
        samples = add_entry_to_sample_distributions(
            samples=samples, constraint_config=constraint_config, varnum=varnum
        )
        ar_distributions[constraint] = {}
        ar_distributions[constraint]["bins"] = np.histogram(
            samples[constraint], bins=100, density=True
        )[1]
        ar_distributions[constraint]["values"] = samples[constraint]

    accepted = pd.DataFrame(
        data_in_dict,
        index=np.arange(targ.shape[0]),
    )
    weights, gofs, gofs_full = calculate_sample_weights(
        ar_distributions, accepted, niterations=60
    )

    if plot_pam_distributions:
        pam_plotting(parammat, name_epithet=f"post1{file_endstring}")
        pam_plotting(
            parammat,
            weights=np.minimum(weights, 1),
            name_epithet=f"post2{file_endstring}",
        )
    effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
    print("Number of effective samples:", effective_samples)

    assert effective_samples >= output_ensemble_size

    draws = []
    drawn_samples = accepted.sample(
        n=output_ensemble_size, replace=False, weights=weights, random_state=10099
    )
    print(drawn_samples)

    output_dir = os.path.join(get_project_root(), "output")

    sample_ids = np.load(
        os.path.join(output_dir, f"valid_sample_ids_all_chunks{file_endstring}.npy"),
        allow_pickle=True,
    )[drawn_samples.index.to_list()]
    make_config_distro_json(
        parammat.iloc[drawn_samples.index.to_list(), :].to_numpy().transpose(),
        parammat.columns,
        f"draw_samples_{output_ensemble_size}{file_endstring}.json",
        index_list=sample_ids,
    )


# Chris plotting stuff, not needed
"""
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

"""

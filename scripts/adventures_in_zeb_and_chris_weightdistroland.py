import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from tqdm.auto import tqdm



def plot_weights(bins, weights,weights_plots_num, unique_code):
    plt.scatter(bins, weights)
    plt.title(f"Weight plot number {weights_plots_num} last added {unique_code}")
    plt.savefig(f"weight_plot_{weights_plots_num}_uncorr.png")
    plt.clf()


def calculate_sample_weights(distributions, samples, niterations=50):
    weights_plots_num = 0
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in tqdm(
        range(niterations), desc="Iterations", leave=False):
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
            print(our_values_bin_idx)
            plot_weights(weights=weights, bins=samples[unique_code][our_values_bin_idx], weights_plots_num=weights_plots_num, unique_code=unique_code)
            weights_plots_num = weights_plots_num + 1

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
    #print(samples.columns)
    #print(samples[unique_code])
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

accepted = pd.DataFrame(
    {
        "constraint_1": scipy.stats.norm.rvs(
            loc=0, scale=1, size=10**5, random_state=70173
        ),
        "constraint_2": scipy.stats.norm.rvs(
            loc=0, scale=1, size=10**5, random_state=74556
        ),
    },
    index=np.arange(10**5),
)
constraints = {
    "constraint_1",
    "constraint_2"
}
samples = {}
samples["constraint_1"] = scipy.stats.norm.rvs(
    loc=-1., scale=1, size=10**5, random_state=18196
)
samples["constraint_2"] = scipy.stats.norm.rvs(
    loc=1, scale=1, size=10**5, random_state=53481
)

#sys.exit(4)
ar_distributions = {}
for constraint in constraints:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
            samples[constraint], bins=100, density=True
        )[1]
    ar_distributions[constraint]["values"] = samples[constraint]


iteration_nums = [1,5,10,20,30,50]
weights, gofs, gofs_full = calculate_sample_weights(
        ar_distributions, accepted, niterations=5
    )
sys.exit(4)
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(30,30))
for i, iteration_num in enumerate(iteration_nums):
    weights, gofs, gofs_full = calculate_sample_weights(
        ar_distributions, accepted, niterations=iteration_num
    )

    effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
    print("Number of effective samples:", effective_samples)
    output_ensemble_size = 500
    assert effective_samples >= output_ensemble_size

    draws = []
    drawn_samples = accepted.sample(
        n=output_ensemble_size, replace=False, weights=weights, random_state=10099
    )
    print(drawn_samples.head())
    #sys.exit(4)
    start = -4
    stop = 4
    x_for_plot = np.linspace(start, stop, 1000)
    target_1 = scipy.stats.gaussian_kde(samples["constraint_1"])
    target_2 = scipy.stats.gaussian_kde(samples["constraint_2"])
    prior_1 = scipy.stats.gaussian_kde(accepted["constraint_1"])
    prior_2 = scipy.stats.gaussian_kde(accepted["constraint_2"])
    post_1 = scipy.stats.gaussian_kde(drawn_samples["constraint_1"])
    post_2 = scipy.stats.gaussian_kde(drawn_samples["constraint_2"])
    axnow_1 = axs[i//3*2, i%3]
    axnow_2 = axs[i//3*2 + 1, i%3]
    axnow_1.plot(x_for_plot, target_1(x_for_plot), label=("target"))
    axnow_1.plot(x_for_plot, prior_1(x_for_plot), label=("prior"))
    axnow_1.plot(x_for_plot, post_1(x_for_plot), label=("posterior"))
    axnow_2.plot(x_for_plot, target_2(x_for_plot), label=("target"))
    axnow_2.plot(x_for_plot, prior_2(x_for_plot), label=("prior"))
    axnow_2.plot(x_for_plot, post_2(x_for_plot), label=("posterior"))
    axnow_2.legend()
    axnow_1.legend()
    axnow_1.set_title(f"Constraint 1 distributions with {iteration_num} iterations")
    axnow_2.set_title(f"Constraint 2 distributions with {iteration_num} iterations")

plt.tight_layout()
plt.savefig("test_2_distributions.png")
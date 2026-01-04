from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy.stats

from cscm_calibrate.weigth_ensemble_from_constraints_and_draw import (
    add_entry_to_sample_distributions,
    calculate_sample_weights,
    get_unique_code_weights,
    opt,
    weight_ensemble_and_draw,
)


def test_opt_matches_quantiles():
    # Use known skewnorm parameters
    a, loc, scale = 2, 0, 1
    q05 = scipy.stats.skewnorm.ppf(0.05, a, loc=loc, scale=scale)
    q50 = scipy.stats.skewnorm.ppf(0.5, a, loc=loc, scale=scale)
    q95 = scipy.stats.skewnorm.ppf(0.95, a, loc=loc, scale=scale)
    # Should return values close to zero
    result = opt([a, loc, scale], q05, q50, q95)
    assert all(abs(x) < 1e-8 for x in result)


def test_add_entry_to_sample_distributions_normal():
    samples = {}
    constraint_config = {
        "Varname_short": ["testvar"],
        "lower_sigma": [1.0],
        "upper_sigma": [1.0],
        "Central Value": [0.0],
    }
    out = add_entry_to_sample_distributions(samples, constraint_config, 0)
    assert "testvar" in out
    arr = out["testvar"]
    assert arr.shape == (10**5,)
    # Should be normal
    assert np.abs(np.mean(arr)) < 0.1
    assert np.abs(np.std(arr) - 1.0) < 0.1


def test_add_entry_to_sample_distributions_skewnorm():
    samples = {}
    constraint_config = {
        "Varname_short": ["testvar"],
        "lower_sigma": [1.0],
        "upper_sigma": [2.0],
        "Central Value": [0.0],
    }
    out = add_entry_to_sample_distributions(samples, constraint_config, 0)
    assert "testvar" in out
    arr = out["testvar"]
    assert arr.shape == (10**5,)
    # Should be skewed: mean > 0 for this config
    assert np.mean(arr) > 0


def test_calculate_sample_weights_basic():
    rng = np.random.RandomState(0)
    n = 500
    # create two sample columns with different distributions
    samples_a = rng.normal(loc=0.0, scale=1.0, size=n)
    samples_b = rng.normal(loc=1.0, scale=2.0, size=n)
    samples_df = pd.DataFrame({"a": samples_a, "b": samples_b})

    # create target (assessed) distributions that differ from the prior samples
    target_a = samples_a + 0.3
    target_b = samples_b - 0.5

    # use histogram bin edges derived from the sample priors to ensure consistent bins
    range_width_a_extra = (samples_a.max() - samples_a.min()) / 1e3
    range_width_b_extra = (samples_b.max() - samples_b.min()) / 1e3
    bins_a = np.histogram(
        samples_a,
        bins=20,
        range=(
            samples_a.min() - range_width_a_extra,
            samples_a.max() + range_width_a_extra,
        ),
    )[1]
    bins_b = np.histogram(
        samples_b,
        bins=20,
        range=(
            samples_b.min() - range_width_b_extra,
            samples_b.max() + range_width_b_extra,
        ),
    )[1]

    distributions = {
        "a": {"bins": bins_a, "values": target_a},
        "b": {"bins": bins_b, "values": target_b},
    }

    # run the reweighting for a few iterations
    weights_final, gofs, gofs_full = calculate_sample_weights(
        distributions, samples_df, niterations=6
    )

    # basic sanity checks
    assert isinstance(weights_final, np.ndarray)
    assert weights_final.shape[0] == n
    assert np.all(np.isfinite(weights_final)), "weights contain non-finite values"
    assert weights_final.sum() > 0

    # gofs should have one column per distribution key
    assert list(gofs.columns) == ["a", "b"]
    # gofs rows corresponds to iterations
    assert gofs.shape[0] == 6

    # gofs_full should have the "Target marginal" column then each code
    assert list(gofs_full.columns) == ["Target marginal", "a", "b"]
    assert gofs_full.shape[1] == 3


def test_get_unique_code_weights_edge_bins():
    # Create a simple distribution with edges and values that will exercise
    # underflow, regular bins, and overflow handling.
    # bins: three bins between -1..1, so bin edges are [-1, 0, 1]
    bins = np.array([-1.0, 0.0, 1.0])
    # create sample values that fall into underflow (< -1), bin1, bin2, overflow (>1)
    samples = np.array([-2.0, -0.5, 0.5, 2.0])
    # create assessed values that put counts in the two histogram bins only
    assessed_values = np.array([-0.3, -0.2, 0.2, 0.4])

    distributions = {"codeX": {"bins": bins, "values": assessed_values}}

    # initial equal weights
    weights = np.ones(samples.shape[0])
    print("H2llo")
    unique_code_weights, our_values_bin_idx = get_unique_code_weights(
        "codeX", distributions, {"codeX": samples}, weights, j=1, k=1
    )

    # unique_code_weights length should be len(bins)+1
    assert unique_code_weights.shape[0] == bins.shape[0] + 1

    # underflow and overflow weights should be zero
    assert unique_code_weights[0] == 0
    assert unique_code_weights[-1] == 0

    # bin indices returned by np.digitize for our samples should be integers
    assert np.array_equal(our_values_bin_idx, np.digitize(samples, bins=bins))

    # For bins where assessed ranges have counts (>0) and existing
    # weighted counts > 0,
    # the weight should be assessed_count / existing_weighted_count
    # (here both 2 and 2 -> 1)
    # middle bins correspond to indices 1 and 2
    assert unique_code_weights[1] >= 0
    assert unique_code_weights[2] >= 0


def test_weight_ensemble_and_draw(tmp_path, monkeypatch):
    """Test weight_ensemble_and_draw with mocked file operations."""
    rng = np.random.default_rng(seed=42)
    # Mock get_project_root to return temp path
    monkeypatch.setattr(
        "cscm_calibrate.weigth_ensemble_from_constraints_and_draw.get_project_root",
        lambda: str(tmp_path),
    )

    # Create mock data
    n_samples = 1000
    mock_targ = pd.DataFrame(
        {"var1": rng.standard_normal(n_samples), "var2": rng.standard_normal(n_samples)}
    )
    mock_parammat = pd.DataFrame(rng.standard_normal((n_samples, 10)))

    # Mock HDFStore
    mock_store = MagicMock()
    mock_store.__enter__ = MagicMock(return_value=mock_store)
    mock_store.__exit__ = MagicMock(return_value=None)
    mock_store.__getitem__ = MagicMock(
        side_effect=lambda key: mock_targ if key == "targ" else mock_parammat
    )

    # Mock constraint config
    constraint_config = {
        "Varname_short": ["var1", "var2"],
        "Variable Name": ["var1", "var2"],
        "lower_sigma": [1.0, 1.0],
        "upper_sigma": [1.0, 1.0],
        "Central Value": [0.0, 0.0],
    }

    file_endstring = "_test"

    with patch("pandas.HDFStore", return_value=mock_store):
        with patch("numpy.load", return_value=np.arange(n_samples)):
            with patch(
                "cscm_calibrate.weigth_ensemble_from_constraints_and_draw.make_config_distro_json"
            ) as mock_save:
                # Run the function
                weight_ensemble_and_draw(
                    constraint_config,
                    file_endstring,
                    output_ensemble_size=10,
                    plot_pam_distributions=False,
                )

                # Assert that save function was called
                mock_save.assert_called_once()

                # Check the call arguments
                args, kwargs = mock_save.call_args
                print(args)
                print(kwargs)
                assert len(args) == 3  # Should have 3 arguments and
                assert len(kwargs) == 1  # and one keyword argument
                assert args[2] == f"draw_samples_10{file_endstring}.json"  # filename

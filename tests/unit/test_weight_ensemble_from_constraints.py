import numpy as np
import scipy.stats

from cscm_calibrate.weigth_ensemble_from_constraints_and_draw import (
    add_entry_to_sample_distributions,
    opt,
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

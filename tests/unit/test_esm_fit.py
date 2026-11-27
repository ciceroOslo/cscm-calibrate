import numpy as np
import pytest
from scipy.stats import multivariate_normal

from cscm_calibrate.esm import fit
from cscm_calibrate.esm.emulator import Emulator, default_params
from cscm_calibrate.esm.targets import Block

TRUTH = default_params() | {"log_sdT": -3, "log_sdN": -2, "rho_d": 0.5, "log_sm": -3}


def test_block_likelihood_is_multivariate_normal():
    rng = np.random.default_rng(0)
    b = Block(np.arange(10), np.zeros(10), 3, 0.1, 0.4)
    r, u = rng.standard_normal(10) * 0.2, np.linspace(0, 1, 10)
    lag = np.abs(np.subtract.outer(b.x, b.x))
    cov = 0.1**2 * 0.4**lag + 0.2**2 * 0.7**lag + 0.05**2 * np.outer(u, u)
    expected = multivariate_normal(np.zeros(10), cov).logpdf(r)
    assert fit.BlockLikelihood(b)(r, 0.2, 0.7, 0.05, u) == pytest.approx(expected)


def test_scaling_basis_is_warming_since_2021_for_tiers():
    x = np.arange(2000, 2031)
    u = fit.scaling_basis("M/T", x, 0.1 * (x - 2000))
    assert (u[x <= 2021] == 0).all()
    assert u[-1] == pytest.approx(0.9)
    assert (fit.scaling_basis("1pctCO2/T", x, x) == x).all()


@pytest.fixture
def synthetic(comps):
    """Targets emulated from TRUTH, with tier "X" renamed to "M"."""
    comps = {k.replace("X/", "M/"): v for k, v in comps.items()}
    o = Emulator(comps, ["M"]).run(TRUTH)
    blocks = {k: Block(x, y, 1, 0.05, 0.5) for k, (x, y) in o.items() if k != "M/N"}
    return blocks, comps


def test_log_posterior_peaks_near_truth(synthetic):
    logp = fit.LogPosterior(*synthetic, ["M"])
    x = np.array([TRUTH[n] for n in fit.NAMES])
    perturbed = x.copy()
    perturbed[fit.NAMES.index("lambda")] *= 1.3
    assert logp(x) > logp(perturbed)
    outside = x.copy()
    outside[fit.NAMES.index("s_co2")] = 2.0
    assert logp(outside) == -np.inf


@pytest.mark.slow
def test_fit_runs(synthetic):
    r = fit.fit(*synthetic, ["M"], workers=2, de_iter=1, nsteps=4, nwalkers=36)
    assert r["samples"].shape == (2 * 36, len(fit.NAMES))
    assert np.isfinite(r["log_prob"]).all()

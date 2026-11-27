import numpy as np
import pandas as pd
import pytest
from ciceroscm import CICEROSCM

from cscm_calibrate.esm import forcing
from cscm_calibrate.esm.emulator import (
    COMPONENTS,
    FIXED,
    I0,
    THERMAL,
    Emulator,
    default_params,
)
from cscm_calibrate.esm.targets import HIST_END

N = forcing.NYEND - forcing.NYSTART + 1


def test_zero_forcing_gives_no_response(comps):
    p = default_params() | {"s_co2": 0, "s_ari": 0, "s_aci": 0, "s_volc": 0}
    o = Emulator(comps, ["X"]).run(p, idealised=False)
    np.testing.assert_allclose(o["X/T"][1], 0, atol=1e-12)
    np.testing.assert_allclose(o["X/N"][1], 0, atol=1e-12)


def test_response_is_linear_in_scalings(comps):
    em = Emulator(comps, ["X"], idealised=False)
    t = {s: em.run(default_params() | {"s_ari": s})["X/T"][1] for s in (0, 1, 2)}
    # near-linear: the upwelling-diffusion model has a weak temperature threshold
    np.testing.assert_allclose(t[2] - t[0], 2 * (t[1] - t[0]), atol=1e-3)
    assert t[1][-1] < t[0][-1]  # aerosol cools


def test_diagnostics(comps):
    p = default_params()
    d = Emulator(comps, ["X"]).diagnostics(p)
    # Gregory ECS is close to lambda * F2x for a near-linear model
    assert d["ECS"] == pytest.approx(p["lambda"] * 7.4 / 2, rel=0.15)
    assert 0 < d["TCR"] < d["ECS"]
    assert d["ERF_aer_2019"] == pytest.approx(-0.65 * 169 / 250)


def test_matches_native_forcing_driven_run(comps):
    """Replacing the forcing per run gives the same result as ciceroscm's own path."""
    p = default_params() | {"s_ari": 1.5, "s_volc": 0.8}
    em = Emulator(comps, ["X"], idealised=False)
    # ciceroscm runs from 1750 with zero forcing before the emulator's 1850 start
    fn, fs, volc_n, volc_s = (
        np.pad(f, [(I0, 0)] + [(0, 0)] * (f.ndim - 1)) for f in em.forcing("X", p)
    )
    years = np.arange(forcing.NYSTART, forcing.NYEND + 1)
    df = pd.DataFrame(
        {"FORC_NH": fn, "FORC_SH": fs, "total": 0.5 * (fn + fs)}, index=years
    )
    scm = CICEROSCM(
        {
            "forc_data": df,
            "nystart": forcing.NYSTART,
            "nyend": forcing.NYEND,
            "emstart": forcing.NYEND + 1,
            "sunvolc": 1,
            "rf_sun_data": np.zeros(N),
            "rf_volc_n_data": volc_n,
            "rf_volc_s_data": volc_s,
            "idtm": 24,
        }
    )
    scm._run({"results_as_dict": True}, pamset_udm={k: p[k] for k in THERMAL} | FIXED)
    t = scm.results["dT_glob_air"][I0:]
    o = em.run(p)
    np.testing.assert_allclose(o["X/T"][1], t - t[em.base].mean(), atol=1e-9)
    np.testing.assert_allclose(o["X/N"][1], scm.results["RIB_glob"][I0:], atol=1e-9)


def test_tiers_sharing_history_match_separate_runs(comps):
    """Branching tiers from one historical run changes nothing."""
    after = np.arange(N) > HIST_END - forcing.NYSTART
    comps |= {f"Y/{k}": comps[f"X/{k}"] for k in COMPONENTS}
    comps["Y/co2"] = np.where(after, 2 * comps["X/co2"], comps["X/co2"])
    comps |= {f"Z/{k}": 1.1 * comps[f"X/{k}"] for k in COMPONENTS}
    em = Emulator(comps, ["X", "Y", "Z"], idealised=False)
    assert em.hist == {"X": "X", "Y": "X", "Z": "Z"}
    p = default_params()
    o = em.run(p)
    for t in "YZ":
        alone = Emulator(comps, [t], idealised=False).run(p)
        for v in "TN":
            np.testing.assert_array_equal(o[f"{t}/{v}"][1], alone[f"{t}/{v}"][1])

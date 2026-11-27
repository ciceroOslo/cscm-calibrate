import numpy as np
import pytest

from cscm_calibrate.esm.forcing import NYEND, NYSTART

N = NYEND - NYSTART + 1


@pytest.fixture
def comps():
    """Synthetic components: CO2 and aerosol ramps after 1850, one eruption."""
    ramp = np.clip(np.arange(N) - 100, 0, None) / 250
    c = {f"X/{k}": np.zeros(N) for k in ("solar", "other_n", "other_s")}
    c |= {"X/co2": 2 * ramp, "X/ari_n": -0.6 * ramp, "X/ari_s": -0.2 * ramp}
    c |= {"X/aci_n": -0.4 * ramp, "X/aci_s": -0.1 * ramp}
    volc = np.zeros((N, 12))
    volc[241] = -3.0  # 1991
    c |= {"X/volc_n": volc, "X/volc_s": volc}
    c["abrupt-4xCO2/co2"] = np.full(150, 7.4)
    c["1pctCO2/co2"] = 7.4 / 140 * np.arange(1, 141)
    return c

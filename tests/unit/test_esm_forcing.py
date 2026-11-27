import os

import numpy as np
import pandas as pd
import pytest

from cscm_calibrate.esm import forcing

YEARS = np.arange(forcing.NYSTART, forcing.NYEND + 1)
I0 = forcing.PIC_YEAR - forcing.NYSTART


def test_picontrol_start_holds_1850_state():
    ramp = pd.DataFrame({"CH4": YEARS * 1.0}, index=YEARS)
    sd = {
        "concentrations_data": ramp,
        "emissions_data": pd.DataFrame(
            {"CO2_FF": 1.0, "CO2_AFOLU": 1.0, "SO2": YEARS * 1.0}, index=YEARS
        ),
        "rf_luc_data": pd.DataFrame(YEARS * 1.0),
        "rf_volc_data": pd.DataFrame(np.full((len(YEARS), 12), -1.0)),
        "rf_sun_data": pd.Series(YEARS * 1.0),
    }
    pic = forcing.picontrol_start(sd)
    assert (pic["concentrations_data"].loc[:1850, "CH4"] == 1850).all()
    assert (pic["emissions_data"].loc[:1849, ["CO2_FF", "CO2_AFOLU"]] == 0).all().all()
    assert (pic["emissions_data"].loc[:1850, "SO2"] == 1850).all()
    assert (pic["rf_luc_data"].iloc[: I0 + 1] == 1850).all().all()
    assert (pic["rf_volc_data"].iloc[:I0] == 0).all().all()
    assert (pic["rf_volc_data"].iloc[I0:] == -1).all().all()
    assert (pic["rf_sun_data"].iloc[:I0] == 0).all()
    assert pic["rf_sun_data"].iloc[I0 : I0 + 24].mean() == pytest.approx(0)
    assert sd["concentrations_data"].loc[1750, "CH4"] == 1750  # input unchanged


@pytest.mark.skipif(
    "CSCM_INDATA" not in os.environ, reason="needs ciceroscm input data"
)
def test_components_add_up():
    c = forcing.scenario_components(os.environ["CSCM_INDATA"], "M")
    split = ("aci", "ari", "other")
    rebuilt = c["co2"] + sum(0.5 * (c[f"{k}_n"] + c[f"{k}_s"]) for k in split)
    np.testing.assert_allclose(rebuilt, c["total"], atol=1e-9)
    for k in ("co2", "total", *(f"{k}_{h}" for k in split for h in "ns")):
        np.testing.assert_allclose(c[k][: I0 + 1], 0, atol=1e-9)
    assert c["ari_n"][2019 - forcing.NYSTART] < c["ari_s"][2019 - forcing.NYSTART] < 0

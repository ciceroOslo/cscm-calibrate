"""
CICERO-SCM forcing components for the CMIP7 ScenarioMIP tiers and idealised runs

Each concentration-driven scenario is run once through the CICERO-SCM
concentrations handler and its forcing is split into components that a
per-ESM fit can scale:

- ``co2``: CO2 ERF (global)
- ``ari_n``, ``ari_s``: aerosol-radiation interaction (SO4 direct, BC, OC)
- ``aci_n``, ``aci_s``: aerosol-cloud interaction (SO4 indirect)
- ``other_n``, ``other_s``: everything else (non-CO2 GHGs, ozone, land use, ...)
- ``volc_n``, ``volc_s``: volcanic, monthly
- ``solar``: solar

``_n``/``_s`` are the northern and southern hemisphere. Runs start from a
constant 1850 state (see :func:`picontrol_start`) and forcing is relative to
1850.

    python -m cscm_calibrate.esm.forcing <indata_dir> <output.npz>
"""

from __future__ import annotations

import sys

import numpy as np
from ciceroscm import CICEROSCM

from cscm_calibrate import shared_functions as shf
from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm

TIERS = ["VL", "LN", "L", "ML", "M", "HL", "H"]
NYSTART, NYEND = 1750, 2100
PIC_YEAR = 1850
GASES = "gases_vupdate_2024_WMO_added_new.txt"
# historical CMIP7 solar, scenario independent and running to 2500
SOLAR = "solar_RCMIP_historical_RCMIP3.txt"
# aerosol forcing is linear in these coefficients, so switching them off isolates it
ACI_OFF = {"qindso2": 0.0}
AER_OFF = {"qdirso2": 0.0, "qindso2": 0.0, "qbc": 0.0, "qoc": 0.0, "qbmb": 0.0}


def picontrol_start(sd: dict) -> dict:
    """
    Hold 1750-1849 at the 1850 state, as an ESM branching from piControl

    Concentrations, non-CO2 emissions and land-use albedo keep their 1850
    values, CO2 emissions and volcanic forcing are zero, and solar is relative
    to its 1850-1873 mean (the CMIP piControl convention).
    """
    i0 = PIC_YEAR - NYSTART
    conc = sd["concentrations_data"].copy()
    conc.loc[conc.index < PIC_YEAR] = conc.loc[PIC_YEAR].to_numpy()
    emis = sd["emissions_data"].copy()
    emis.loc[emis.index < PIC_YEAR] = emis.loc[PIC_YEAR].to_numpy()
    emis.loc[emis.index < PIC_YEAR, ["CO2_FF", "CO2_AFOLU"]] = 0.0
    luc = sd["rf_luc_data"].copy()
    luc.iloc[:i0] = luc.iloc[i0].to_numpy()
    volc = sd["rf_volc_data"].copy()
    volc.iloc[:i0] = 0.0
    sun = sd["rf_sun_data"] - sd["rf_sun_data"].iloc[i0 : i0 + 24].mean()
    sun.iloc[:i0] = 0.0
    return {
        **sd,
        "concentrations_data": conc,
        "emissions_data": emis,
        "rf_luc_data": luc,
        "rf_volc_data": volc,
        "rf_sun_data": sun,
    }


def scenario_scendata(indata: str, tier: str) -> dict:
    """Concentration-driven scendata for a ScenarioMIP tier, from 1850 rest."""
    # ML has no RCMIP3 volcanic or land-use file: use M's volcanic and ML's AR7
    volc = f"VOLC_RCMIP_scen7-{'M' if tier == 'ML' else tier}_RCMIP3.txt"
    luc = f"LUCalbedo_RCMIP_scen7-{tier}_{'AR7' if tier == 'ML' else 'RCMIP3'}.txt"
    scen = f"scen7-{tier}"
    sd = define_scendata_for_scm(
        indata,
        gaspam=GASES,
        df_conc=f"{scen}_conc_{GASES}",
        df_emis=f"{scen}_em_{GASES}",
        nystart=NYSTART,
        nyend=NYEND,
        emstart=NYEND,
        sunvolc=1,
        rf_volc_file=volc,
        rf_solar_file=SOLAR,
        rf_luc_file=luc,
    )[0]
    sd = picontrol_start({**sd, "conc_run": True, "scenname": scen})
    # pass volcanic per hemisphere: ciceroscm (<= b7adfc6) ignores rf_volc_data
    volc = sd.pop("rf_volc_data")
    return {**sd, "rf_volc_n_data": volc, "rf_volc_s_data": volc}


def _hemispheric_run(model: CICEROSCM, pamset_emiconc: dict | None = None):
    """Run once, recording the hemispheric forcing passed to the thermal model."""
    rec = []
    conc2forc = model.ce_handler.conc2forc

    def recording(*args, **kwargs):
        out = conc2forc(*args, **kwargs)
        rec.append(out[:3])
        return out

    model.ce_handler.conc2forc = recording
    try:
        model._run({"results_as_dict": True}, pamset_emiconc=pamset_emiconc)
    finally:
        model.ce_handler.conc2forc = conc2forc
    total, nh, sh = (np.array(x, dtype=float) for x in zip(*rec))
    return total, nh, sh, model.results["forcing"]["CO2"].to_numpy()


def scenario_components(indata: str, tier: str) -> dict[str, np.ndarray]:
    """Split the forcing of one tier into components (1750-2100, relative to 1850)."""
    model = CICEROSCM(scenario_scendata(indata, tier))
    n = NYEND - NYSTART + 1
    solar = np.asarray(model._sun_arr, dtype=float)[:n]
    total, nh, sh, co2 = _hemispheric_run(model)
    _, nh_noaci, sh_noaci, _ = _hemispheric_run(model, ACI_OFF)
    _, nh_noaer, sh_noaer, _ = _hemispheric_run(model, AER_OFF)
    c = {
        "co2": co2,
        "aci_n": nh - nh_noaci,
        "aci_s": sh - sh_noaci,
        "ari_n": nh_noaci - nh_noaer,
        "ari_s": sh_noaci - sh_noaer,
        # the handler's hemispheric forcing includes solar and CO2
        "other_n": nh_noaer - solar - co2,
        "other_s": sh_noaer - solar - co2,
        "total": total,  # global, excluding solar and volcanic
    }
    c = {k: v - v[PIC_YEAR - NYSTART] for k, v in c.items()}
    c["solar"] = solar
    c["volc_n"] = np.asarray(model._volc_n_arr, dtype=float)[:n]
    c["volc_s"] = np.asarray(model._volc_s_arr, dtype=float)[:n]
    return c


def idealised_co2(indata: str, exp: str, nyend: int = NYEND) -> np.ndarray:
    """CO2 ERF of an idealised CO2 experiment, from its first perturbed year."""
    zeros = {
        f"df_nat_{g.lower()}": shf.make_dataframe_of_zeros(g, NYSTART, nyend + 1)
        for g in ("CH4", "N2O")
    }
    sd = define_scendata_for_scm(
        indata,
        gaspam=GASES,
        df_conc=f"{exp}_conc_{GASES}",
        df_emis=f"esm-piControl_em_{GASES}",
        rf_luc_file="LUCalbedo_RCMIP_constant_zero_RCMIP3.txt",
        nystart=NYSTART,
        emstart=nyend + 1,
        nyend=nyend,
        sunvolc=0,
        **zeros,
    )[0]
    sd.update(conc_run=True, scenname=exp, ref_yr=min(nyend - 1, 2010))
    model = CICEROSCM(sd)
    model._run({"results_as_dict": True})
    co2 = model.results["forcing"]["CO2"].to_numpy()
    return co2[np.argmax(co2.round(3) != 0) :]


def build(indata: str) -> dict[str, np.ndarray]:
    """Build components for all tiers (``"<tier>/<component>"``) and idealised runs."""
    data = {
        f"{tier}/{k}": v
        for tier in TIERS
        for k, v in scenario_components(indata, tier).items()
    }
    for exp in ("abrupt-4xCO2", "1pctCO2"):
        data[f"{exp}/co2"] = idealised_co2(indata, exp)
    return data


if __name__ == "__main__":
    np.savez(sys.argv[2], **build(sys.argv[1]))

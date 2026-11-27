"""SCM forcing components for CMIP7 ScenarioMIP tiers and idealised CO2 runs.

Each concentration-driven scenario is run once through the CICERO-SCM
emissions/concentrations handler. Every call to ``conc2forc`` is recorded to get the
hemispheric forcing, and it is split into components that the per-ESM fit can scale:

  co2    CO2 ERF (global, hemispherically symmetric)
  ari    aerosol-radiation (SO4 direct + BC + OC), NH/SH
  aci    aerosol-cloud (SO4 indirect), NH/SH
  other  everything else from the handler (non-CO2 GHGs, O3, H2O, land-use albedo...), NH/SH
  volc   volcanic, NH/SH monthly (applied by the thermal model)
  solar  solar (added by the thermal driver)

Aerosol terms are linear in their q-coefficients, so the hemispheric aerosol forcing is
recovered by differencing runs with the aerosol q's switched off.

    python scripts/cmip7_esm/scm_forcing.py      # writes output/cmip7_esm/forcing_components_pic.npz
    CMIP7_BASELINE=1750 python ...                # the standard 1750-start cache (pilot v1/v2)
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CICEROSCM = Path("/storage/no-backup-nac/users/bensan/cscm-calibrate/sst_hash_pattern_test/runA/ciceroscm")
INDATA = "/storage/no-backup-nac/users/bensan/cscm-calibrate/sst_hash_pattern_test/runA/indata/"
GASES = "gases_vupdate_2024_WMO_added_new.txt"
sys.path.insert(0, str(CICEROSCM / "src"))
sys.path.insert(0, str(ROOT / "src"))

from ciceroscm import CICEROSCM as _CSCM  # noqa: E402
from cscm_calibrate import shared_functions as shf  # noqa: E402
from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm  # noqa: E402

OUT = ROOT / "output" / "cmip7_esm"
TIERS = ["VL", "LN", "L", "ML", "M", "HL", "H"]
NYSTART, NYEND = 1750, 2100
PIC_YEAR = 1850
# "pic": pre-1850 held at the 1850 state and forcing relative to 1850 (ESM-like piControl start);
# "1750": the SCM's standard 1750 start (pilot v1/v2)
BASELINE = os.environ.get("CMIP7_BASELINE", "pic")
# scen7 solar links are broken; the CMIP7 historical solar file runs to 2500 and is
# scenario independent. ML has no RCMIP3 volcanic/LUC files: use M's volcanic, ML's AR7 LUC.
SOLAR = "solar_RCMIP_historical_RCMIP3.txt"


def _volc(tier):
    return f"VOLC_RCMIP_scen7-{'M' if tier == 'ML' else tier}_RCMIP3.txt"


def _luc(tier):
    return f"LUCalbedo_RCMIP_scen7-{tier}_{'AR7' if tier == 'ML' else 'RCMIP3'}.txt"


AER_OFF = {"qdirso2": 0.0, "qindso2": 0.0, "qbc": 0.0, "qoc": 0.0, "qbmb": 0.0}
ACI_OFF = {"qindso2": 0.0}


def picontrol_start(sd):
    """Make 1750-1849 a constant 1850 pre-industrial state, as for an ESM branching from piControl.

    Concentrations, non-CO2 emissions and land-use albedo are held at their 1850 values, CO2
    emissions are zero (so CO2 stays at its 1850 value, which becomes the carbon cycle's
    pre-industrial reference), volcanic forcing is zero, and solar is expressed relative to
    its 1850-1873 mean (the CMIP piControl solar convention). Forcing that is constant before
    1850 is removed by the users of this data (subtracting the 1850 value).
    """
    i0 = PIC_YEAR - NYSTART
    sd = dict(sd)
    conc = sd["concentrations_data"].copy()
    conc.loc[conc.index < PIC_YEAR] = conc.loc[PIC_YEAR].to_numpy()
    emis = sd["emissions_data"].copy()
    emis.loc[emis.index < PIC_YEAR] = emis.loc[PIC_YEAR].to_numpy()
    emis.loc[emis.index < PIC_YEAR, ["CO2_FF", "CO2_AFOLU"]] = 0.0
    luc = sd["rf_luc_data"].copy()
    luc.iloc[:i0] = luc.iloc[i0].to_numpy()
    volc = sd["rf_volc_data"].copy()
    volc.iloc[:i0] = 0.0  # (handed to ciceroscm as rf_volc_n/s_data by scenario_scendata)
    sun = sd["rf_sun_data"].copy()
    sun = sun - sun.iloc[i0:i0 + 24].mean()
    sun.iloc[:i0] = 0.0
    sd.update(concentrations_data=conc, emissions_data=emis, rf_luc_data=luc, rf_volc_data=volc, rf_sun_data=sun)
    return sd


def scenario_scendata(tier, baseline=None):
    baseline = baseline or BASELINE
    scen = f"scen7-{tier}"
    sd = define_scendata_for_scm(
        INDATA, gaspam=GASES, df_conc=f"{scen}_conc_{GASES}", df_emis=f"{scen}_em_{GASES}",
        nystart=NYSTART, nyend=NYEND, emstart=NYEND, sunvolc=1,
        rf_volc_file=_volc(tier), rf_solar_file=SOLAR, rf_luc_file=_luc(tier),
    )[0]
    assert sd["rf_sun_data"] is not None and sd["rf_volc_data"] is not None, tier
    sd.update(conc_run=True, scenname=scen)
    sd = picontrol_start(sd) if baseline == "pic" else sd
    # ciceroscm's InputHandler drops an "rf_volc_data" key (only rf_volc_n/s_data are accepted)
    # and silently falls back to its default IPCC volcanic series; pass the RCMIP3 data explicitly
    volc = sd.pop("rf_volc_data")
    sd["rf_volc_n_data"] = volc
    sd["rf_volc_s_data"] = volc
    return sd


def _recorded_run(model, pamset_emiconc=None):
    """Run once, returning (tot, fn, fs) per year exactly as passed to the thermal model."""
    rec = []
    orig = model.ce_handler.conc2forc

    def wrapped(*a, **k):
        out = orig(*a, **k)
        rec.append(out[:3])
        return out

    model.ce_handler.conc2forc = wrapped
    try:
        model._run({"results_as_dict": True}, pamset_emiconc=pamset_emiconc)
    finally:
        model.ce_handler.conc2forc = orig
    tot, fn, fs = (np.array(x, dtype=float) for x in zip(*rec))
    return tot, fn, fs, model.results["forcing"].copy()


def scenario_components(tier):
    model = _CSCM(scenario_scendata(tier))
    sun = np.asarray(model._sun_arr, dtype=float)[: NYEND - NYSTART + 1]
    base = _recorded_run(model)
    no_aci = _recorded_run(model, ACI_OFF)
    no_aer = _recorded_run(model, AER_OFF)
    co2 = base[3]["CO2"].to_numpy()
    c = {
        "year": np.arange(NYSTART, NYEND + 1),
        "co2": co2,
        "aci_n": base[1] - no_aci[1], "aci_s": base[2] - no_aci[2],
        "ari_n": no_aci[1] - no_aer[1], "ari_s": no_aci[2] - no_aer[2],
        # remaining handler forcing with solar and CO2 removed
        "other_n": no_aer[1] - sun - co2, "other_s": no_aer[2] - sun - co2,
        "solar": sun,
        "volc_n": np.asarray(model._volc_n_arr, dtype=float)[: NYEND - NYSTART + 1],
        "volc_s": np.asarray(model._volc_s_arr, dtype=float)[: NYEND - NYSTART + 1],
        "total_global": base[0],  # the handler total excludes solar
    }
    if BASELINE == "pic":  # forcing relative to the constant 1850 pre-industrial state
        i0 = PIC_YEAR - NYSTART
        for k in ("co2", "aci_n", "aci_s", "ari_n", "ari_s", "other_n", "other_s", "total_global"):
            assert np.ptp(c[k][: i0 + 1]) < 1e-9, f"{k} not constant before {PIC_YEAR}"
            c[k] = c[k] - c[k][i0]
    # consistency: components rebuild the handler's global total
    rebuilt = co2 + 0.5 * (c["aci_n"] + c["aci_s"] + c["ari_n"] + c["ari_s"] + c["other_n"] + c["other_s"])
    err = np.abs(rebuilt - c["total_global"]).max()
    f = base[3]
    erf_aer = (f["SO4_DIR"] + f["SO4_IND"] + f["BC"] + f["OC"]).to_numpy()
    aer_from_hemi = 0.5 * (c["aci_n"] + c["aci_s"] + c["ari_n"] + c["ari_s"])
    print(f"{tier}: rebuild err {err:.2e}; aerosol vs handler output {np.abs(aer_from_hemi - erf_aer).max():.2e}; "
          f"ERF2019 co2 {co2[2019 - NYSTART]:.2f} aer {erf_aer[2019 - NYSTART]:.2f} total {c['total_global'][2019 - NYSTART]:.2f}",
          flush=True)
    return c


def idealised_co2(exp, nyend):
    """CO2 ERF time series for an idealised concentration experiment (all else pre-industrial)."""
    sd = define_scendata_for_scm(
        INDATA, gaspam=GASES,
        df_nat_ch4=shf.make_dataframe_of_zeros("CH4", NYSTART, nyend + 1),
        df_nat_n2o=shf.make_dataframe_of_zeros("N2O", NYSTART, nyend + 1),
        df_conc=f"{exp}_conc_{GASES}", df_emis=f"esm-piControl_em_{GASES}",
        rf_luc_file="LUCalbedo_RCMIP_constant_zero_RCMIP3.txt",
        nystart=NYSTART, emstart=nyend + 1, nyend=nyend, sunvolc=0,
    )[0]
    sd.update(conc_run=True, scenname=exp, ref_yr=int(min(nyend - 1, 2010)))
    model = _CSCM(sd)
    model._run({"results_as_dict": True})
    f = model.results["forcing"]
    co2 = f["CO2"].to_numpy()
    start = int(np.argmax(np.abs(co2) > 1e-3))  # first perturbed year = experiment year 1
    nonco2 = (f["Total_forcing"] - f["CO2"]).abs().max()
    print(f"{exp}: starts at model year {NYSTART + start}, non-CO2 max {nonco2:.2e}, "
          f"CO2 ERF yr1 {co2[start]:.2f} yr70 {co2[min(start + 69, len(co2) - 1)]:.2f}", flush=True)
    return co2[start:]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for tier in TIERS:
        for k, v in scenario_components(tier).items():
            data[f"{tier}/{k}"] = v
    data["abrupt-4xCO2/co2"] = idealised_co2("abrupt-4xCO2", 2100)
    data["1pctCO2/co2"] = idealised_co2("1pctCO2", 2100)
    np.savez(_cache(), **data)
    print("wrote", _cache())


def _cache():
    return OUT / ("forcing_components.npz" if BASELINE == "1750" else f"forcing_components_{BASELINE}.npz")


def load():
    d = np.load(_cache())
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build()

"""
Forcing-driven CICERO-SCM with per-ESM forcing scalings

The thermal model (upwelling-diffusion, with ocean heat-uptake efficacy) is
driven by the forcing components of :mod:`cscm_calibrate.esm.forcing`, each
multiplied by a scaling. Only the thermal model runs per evaluation (about
0.6 ms per model year).
"""

from __future__ import annotations

import numpy as np
from ciceroscm import CICEROSCM

from cscm_calibrate.esm.forcing import NYEND, NYSTART, PIC_YEAR
from cscm_calibrate.esm.targets import N_ABRUPT

# name: (lower, upper) bounds of the uniform prior. Thermal ranges widen the
# +epsilon prior (tests/test-data/config_file_no_deltaaero_wide_lambda.json)
# where pilot fits reached its bounds.
THERMAL = {
    "lambda": (0.3, 2.5),  # climate sensitivity parameter, K / (W m-2)
    "rlamdo": (1.0, 30.0),  # air-sea heat exchange
    "akapa": (0.05, 1.5),  # vertical diffusivity
    "cpi": (0.1, 1.0),  # polar parameter
    "W": (0.3, 5.0),  # upwelling velocity
    "beto": (0.0, 7.0),  # hemispheric ocean heat exchange
    "mixed": (20.0, 150.0),  # mixed-layer depth, m
    "ocean_efficacy": (0.8, 2.0),  # ocean heat-uptake efficacy
}
SCALINGS = {
    "s_co2": (0.7, 1.4),  # CO2 ERF (the ESM's F2x relative to the SCM's)
    "s_ari": (0.0, 6.0),  # aerosol-radiation interaction
    "s_aci": (0.0, 3.0),  # aerosol-cloud interaction
    "s_other": (0.6, 1.4),  # all other anthropogenic forcing
    "s_volc": (0.2, 1.5),  # volcanic
}
PARAMS = {**THERMAL, **SCALINGS}
FIXED = {"threstemp": 7.0, "lm": 40, "ldtime": 12, "delta_lambda_aero": 0.0}
BASE = (1850, 1900)  # GSAT anomaly reference period for the scenarios


class _ForcingRun:
    """One CICEROSCM instance whose forcing is replaced before each run."""

    def __init__(self, nystart: int, nyears: int):
        self.nystart = nystart
        self.model = CICEROSCM(
            {
                "forc_data": np.zeros(nyears),
                "nystart": nystart,
                "nyend": nystart + nyears - 1,
                "emstart": nystart + nyears,
                "sunvolc": 0,
                "idtm": 24,
                "scenname": "esm",
            }
        )
        # ciceroscm reads forcing year by year through forc_set
        self.model.forc_set = self._forc_set

    def _forc_set(self, yr):
        i = yr - self.nystart
        return self.fn[i], self.fs[i], 0.5 * (self.fn[i] + self.fs[i]), 0.0

    def run(self, pam, fn, fs, volc_n=None, volc_s=None):
        self.fn, self.fs = fn, fs
        m = self.model
        if volc_n is not None:
            m._volc_n_arr, m._volc_s_arr = volc_n, volc_s
            m._volc_n_mean, m._volc_s_mean = volc_n.mean(axis=1), volc_s.mean(axis=1)
        m._run({"results_as_dict": True}, pamset_udm={**pam, **FIXED})
        return m.results


class Emulator:
    """
    Emulate an ESM's scenario and idealised runs for a parameter set

    ``comps`` is the output of :func:`cscm_calibrate.esm.forcing.build`.
    """

    def __init__(self, comps: dict, tiers: list[str], idealised: bool = True):
        self.c = comps
        self.years = np.arange(NYSTART, NYEND + 1)
        self.base = (self.years >= BASE[0]) & (self.years <= BASE[1])
        self.scen = {t: _ForcingRun(NYSTART, len(self.years)) for t in tiers}
        self.ideal = {}
        if idealised:
            for exp in ("abrupt-4xCO2", "1pctCO2"):
                f = comps[f"{exp}/co2"]
                self.ideal[exp] = (_ForcingRun(PIC_YEAR, len(f)), f)

    def forcing(self, tier: str, p: dict):
        """Scaled NH and SH forcing (W m-2) and monthly volcanic forcing."""
        hemi = []
        for h in "ns":
            f = p["s_co2"] * self.c[f"{tier}/co2"] + self.c[f"{tier}/solar"]
            for k in ("ari", "aci", "other"):
                f = f + p[f"s_{k}"] * self.c[f"{tier}/{k}_{h}"]
            hemi.append(f)
        volc = [p["s_volc"] * self.c[f"{tier}/volc_{h}"] for h in "ns"]
        return (*hemi, *volc)

    def run(self, p: dict, tiers=None, idealised: bool = True) -> dict:
        """
        GSAT (T, K) and net TOA flux (N, W m-2) for each experiment

        Returns ``{"<tier>/<var>": (x, y)}`` with x the year for scenarios
        (T relative to 1850-1900) and the year since branching for the
        idealised runs (relative to piControl).
        """
        pam = {k: float(p[k]) for k in THERMAL}
        out = {}
        for tier in self.scen if tiers is None else tiers:
            r = self.scen[tier].run(pam, *self.forcing(tier, p))
            t = r["dT_glob_air"]
            out[f"{tier}/T"] = (self.years, t - t[self.base].mean())
            out[f"{tier}/N"] = (self.years, r["RIB_glob"].copy())
        for exp, (e, f) in self.ideal.items() if idealised else ():
            r = e.run(pam, p["s_co2"] * f, p["s_co2"] * f)
            x = np.arange(1, len(f) + 1)
            out[f"{exp}/T"] = (x, r["dT_glob_air"].copy())
            out[f"{exp}/N"] = (x, r["RIB_glob"].copy())
        return out

    def diagnostics(self, p: dict) -> dict:
        """
        ECS, TCR and 2019 aerosol ERF, as diagnosed for the ESMs

        ECS is from a Gregory regression over abrupt-4xCO2 years 1-150, TCR is
        the 1pctCO2 warming averaged over years 61-80.
        """
        o = self.run(p, tiers=[])
        t4x, n4x = (o[f"abrupt-4xCO2/{v}"][1][:N_ABRUPT] for v in "TN")
        slope, f4x = np.polyfit(t4x, n4x, 1)
        tier, i = next(iter(self.scen)), 2019 - NYSTART
        aer = sum(
            p[f"s_{k}"]
            * 0.5
            * (self.c[f"{tier}/{k}_n"][i] + self.c[f"{tier}/{k}_s"][i])
            for k in ("ari", "aci")
        )
        return {
            "ECS": -f4x / slope / 2,
            "TCR": o["1pctCO2/T"][1][60:80].mean(),
            "ERF_aer_2019": aer,
        }


def default_params() -> dict:
    """Central +epsilon thermal parameters with unit forcing scalings."""
    return {
        "lambda": 0.8,
        "rlamdo": 15.0,
        "akapa": 0.66,
        "cpi": 0.21,
        "W": 2.2,
        "beto": 6.9,
        "mixed": 107.0,
        "ocean_efficacy": 1.0,
        **dict.fromkeys(SCALINGS, 1.0),
    }

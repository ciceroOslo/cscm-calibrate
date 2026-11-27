"""
Forcing-driven CICERO-SCM with per-ESM forcing scalings

The thermal model (upwelling-diffusion, with ocean heat-uptake efficacy) is
driven by the forcing components of :mod:`cscm_calibrate.esm.forcing`, each
multiplied by a scaling. Only the thermal model runs per evaluation (about
0.25 ms per model year), and only from 1850: forcing is zero before then.
Tiers with the same historical forcing share one run to 2021.
"""

from __future__ import annotations

import copy

import numpy as np
from ciceroscm.thermal_model.upwelling_diffusion_model import (
    UpwellingDiffusionModel,
)

from cscm_calibrate.esm.forcing import NYEND, NYSTART, PIC_YEAR
from cscm_calibrate.esm.targets import HIST_END, N_ABRUPT

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
I0 = PIC_YEAR - NYSTART  # index of 1850 in the forcing components
NHIST = HIST_END - PIC_YEAR + 1  # years 1850-2021, shared by the tiers
N_IDEAL = 150  # years of abrupt-4xCO2 and 1pctCO2 emulated
COMPONENTS = ["co2", "solar", "volc_n", "volc_s"] + [
    f"{k}_{h}" for k in ("ari", "aci", "other") for h in "ns"
]


def _integrate(udm, fn, fs, volc_n, volc_s):
    """Step ``udm`` through the given years; returns GSAT and TOA flux."""
    t, n = np.empty(len(fn)), np.empty(len(fn))
    for i in range(len(fn)):
        r = udm.energy_budget([fn[i], fs[i], volc_n[i], volc_s[i]])
        t[i], n[i] = r["dtemp_air"], r["RIB"]
    return t, n


class Emulator:
    """
    Emulate an ESM's scenario and idealised runs for a parameter set

    ``comps`` is the output of :func:`cscm_calibrate.esm.forcing.build`.
    """

    def __init__(self, comps: dict, tiers: list[str], idealised: bool = True):
        for k in COMPONENTS:
            for t in tiers:
                if np.any(comps[f"{t}/{k}"][:I0]):
                    msg = f"{t}/{k}: nonzero forcing before {PIC_YEAR}"
                    raise ValueError(msg)
        self.c = comps
        self.years = np.arange(PIC_YEAR, NYEND + 1)
        self.base = (self.years >= BASE[0]) & (self.years <= BASE[1])
        self.tiers = list(tiers)
        # the first tier with the same historical forcing runs the history
        self.hist = {}
        for t in tiers:
            self.hist[t] = next(
                (u for u in self.hist.values() if self._same_history(t, u)), t
            )
        self.ideal = idealised

    def _same_history(self, a: str, b: str) -> bool:
        n = I0 + NHIST
        return all(
            np.allclose(self.c[f"{a}/{k}"][:n], self.c[f"{b}/{k}"][:n], 0, 1e-12)
            for k in COMPONENTS
        )

    def forcing(self, tier: str, p: dict):
        """Scaled NH and SH forcing (W m-2) and monthly volcanic, from 1850."""
        c = {k: self.c[f"{tier}/{k}"][I0:] for k in COMPONENTS}
        hemi = []
        for h in "ns":
            f = p["s_co2"] * c["co2"] + c["solar"]
            for k in ("ari", "aci", "other"):
                f = f + p[f"s_{k}"] * c[f"{k}_{h}"]
            hemi.append(f)
        return (*hemi, p["s_volc"] * c["volc_n"], p["s_volc"] * c["volc_s"])

    def run(self, p: dict, tiers=None, idealised: bool = True) -> dict:
        """
        GSAT (T, K) and net TOA flux (N, W m-2) for each experiment

        Returns ``{"<tier>/<var>": (x, y)}`` with x the year for scenarios
        (T relative to 1850-1900) and the year since branching for the
        idealised runs (relative to piControl).
        """
        pam = {**{k: float(p[k]) for k in THERMAL}, **FIXED}
        out, hist = {}, {}
        for tier in self.tiers if tiers is None else tiers:
            f = self.forcing(tier, p)
            h = self.hist[tier]
            if h not in hist:
                udm = UpwellingDiffusionModel(pam)
                hist[h] = (udm, _integrate(udm, *(x[:NHIST] for x in f)))
            udm, (th, nh) = hist[h]
            # continue from a copy of the 2021 ocean state
            tf, nf = _integrate(copy.deepcopy(udm), *(x[NHIST:] for x in f))
            t = np.concatenate([th, tf])
            out[f"{tier}/T"] = (self.years, t - t[self.base].mean())
            out[f"{tier}/N"] = (self.years, np.concatenate([nh, nf]))
        no_volc = np.zeros((N_IDEAL, 1))
        for exp in ("abrupt-4xCO2", "1pctCO2") if self.ideal and idealised else ():
            f = p["s_co2"] * self.c[f"{exp}/co2"][:N_IDEAL]
            x = np.arange(1, len(f) + 1)
            udm = UpwellingDiffusionModel(pam)
            tn = _integrate(udm, f, f, no_volc, no_volc)
            out[f"{exp}/T"], out[f"{exp}/N"] = ((x, y) for y in tn)
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
        tier, i = self.tiers[0], 2019 - NYSTART
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

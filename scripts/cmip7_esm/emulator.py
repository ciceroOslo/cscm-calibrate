"""Forcing-driven CICERO-SCM (+epsilon structure) with per-ESM forcing scalings.

theta = thermal parameters of the upwelling-diffusion model (with ocean heat-uptake
efficacy, delta_lambda_aero = 0) + scalings of the SCM's own forcing components
(CO2, aerosol-radiation, aerosol-cloud, other, volcanic) + structural-error terms.

Each experiment is a forcing-driven CICEROSCM instance built once; per evaluation only
the thermal model runs, with hemispheric forcing assembled from the cached components
(scm_forcing.py).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scm_forcing as SF  # noqa: E402
from ciceroscm import CICEROSCM as _CSCM  # noqa: E402

# name, lower, upper (uniform prior). Thermal ranges start from the +epsilon prior config
# (config_file_no_deltaaero_wide_lambda.json); v2 widens those the pilot fits ran into
# (rlamdo, mixed, cpi, W, akapa, s_ari).
PARAMS = [
    ("lambda", 0.3, 2.5),        # climate sensitivity parameter, K per W m-2
    ("rlamdo", 1.0, 30.0),       # air-sea heat exchange (pilot: 5-25)
    ("akapa", 0.05, 1.5),        # vertical diffusivity (pilot: 0.1-0.9)
    ("cpi", 0.1, 1.0),           # polar parameter (pilot: 0.161-0.569)
    ("W", 0.3, 5.0),             # upwelling velocity (pilot: 0.55-2.55)
    ("beto", 0.0, 7.0),
    ("mixed", 20.0, 150.0),      # mixed-layer depth (pilot: 50-150)
    ("ocean_efficacy", 0.8, 2.0),
    ("a_state", -0.3, 0.15),     # feedback state dependence d(alpha)/dT, W m-2 K-2 (0 in +eps)
    ("s_co2", 0.7, 1.4),         # CO2 ERF scaling (per-ESM F2x)
    ("s_ari", 0.0, 6.0),         # aerosol-radiation scaling (pilot: 0-4)
    ("s_aci", 0.0, 3.0),         # aerosol-cloud scaling
    ("s_other", 0.6, 1.4),       # remaining anthropogenic ERF
    ("s_volc", 0.2, 1.5),        # volcanic efficacy/scaling
    # discrepancy model (fit_esm.py): persistent AR(1) term per block + warming-proportional term
    ("log_sdT", -6.0, 0.0),      # AR(1) discrepancy sd, K (log)
    ("log_sdN", -6.0, 0.5),      # AR(1) discrepancy sd, W m-2 (log)
    ("rho_d", 0.0, 0.995),       # discrepancy lag-1 autocorrelation
    ("log_sm", -7.0, -0.7),      # fractional (per-block) warming scaling error (log)
]
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[1] for p in PARAMS])
HI = np.array([p[2] for p in PARAMS])
THERMAL = NAMES[:8]
FIXED_UDM = {"threstemp": 7.0, "lm": 40, "ldtime": 12}
T_VAR = "dT_glob_air"  # GSAT

# structural variants: which parameters are fixed (value) rather than sampled
VARIANTS = {"eps": {"a_state": 0.0}, "eps_state": {}}


class Space:
    """Map between the sampled (free) parameters of a variant and the full theta."""

    def __init__(self, variant="eps"):
        self.variant = variant
        self.fixed = VARIANTS[variant]
        self.free = [i for i, n in enumerate(NAMES) if n not in self.fixed]
        self.lo, self.hi = LO[self.free], HI[self.free]
        self.names = [NAMES[i] for i in self.free]

    def full(self, free_theta):
        th = np.array([self.fixed.get(n, np.nan) for n in NAMES], dtype=float)
        th[self.free] = free_theta
        return th


def pamset_udm(theta):
    p = dict(zip(THERMAL, map(float, theta[: len(THERMAL)])))
    p.update(FIXED_UDM)
    # state-dependent feedback through the thermal model's pattern-feedback hook:
    # alpha_eff = 1/lambda + delta_lambda_aero * w_aero, with w_aero = last year's GSAT
    p["delta_lambda_aero"] = float(theta[NAMES.index("a_state")])
    return p


class _Exp:
    """One forcing-driven experiment: fixed CICEROSCM instance, forcing rebuilt per call."""

    def __init__(self, nystart, nyears):
        self.nystart, self.n = nystart, nyears
        self.m = _CSCM({"forc_data": np.zeros(nyears), "nystart": nystart, "nyend": nystart + nyears - 1,
                        "emstart": nystart + nyears, "sunvolc": 0, "idtm": 24, "scenname": "fit"})
        self.fn = self.fs = np.zeros(nyears)
        self.m.forc_set = self._forc_set

    def _forc_set(self, yr):
        i = yr - self.nystart
        # w_aero carries the previous year's global surface temperature (state-dependent feedback)
        w = self.m.results["dT_glob"][i - 1] if i > 0 else 0.0
        return self.fn[i], self.fs[i], 0.5 * (self.fn[i] + self.fs[i]), w

    def run(self, pam, fn, fs, volc_n=None, volc_s=None):
        self.fn, self.fs = fn, fs
        if volc_n is not None:
            self.m._volc_n_arr, self.m._volc_s_arr = volc_n, volc_s
            self.m._volc_n_mean, self.m._volc_s_mean = volc_n.mean(axis=1), volc_s.mean(axis=1)
        self.m._run({"results_as_dict": True}, pamset_udm=pam)
        if pam["delta_lambda_aero"] != 0.0 and self.m.results["dynamic_lambda"].min() < 0.05:
            raise FloatingPointError("feedback parameter fell below 0.05 W m-2 K-1")
        return self.m.results


class Emulator:
    def __init__(self, tiers, idealised=True, comps=None):
        self.c = comps or SF.load()
        self.tiers = list(tiers)
        ny = SF.NYEND - SF.NYSTART + 1
        self.scen = {t: _Exp(SF.NYSTART, ny) for t in self.tiers}
        self.years = np.arange(SF.NYSTART, SF.NYEND + 1)
        self.base = (self.years >= 1850) & (self.years <= 1900)
        self.ideal = {}
        if idealised:
            for exp in ("abrupt-4xCO2", "1pctCO2"):
                f = self.c[f"{exp}/co2"]
                self.ideal[exp] = (_Exp(1850, len(f)), f)

    def scenario_forcing(self, tier, s):
        g = lambda k: self.c[f"{tier}/{k}"]  # noqa: E731
        co2 = s["s_co2"] * g("co2")
        fn = co2 + s["s_ari"] * g("ari_n") + s["s_aci"] * g("aci_n") + s["s_other"] * g("other_n") + g("solar")
        fs = co2 + s["s_ari"] * g("ari_s") + s["s_aci"] * g("aci_s") + s["s_other"] * g("other_s") + g("solar")
        return fn, fs, s["s_volc"] * g("volc_n"), s["s_volc"] * g("volc_s")

    def run(self, theta, tiers=None, idealised=True):
        """Return {block_key: (x, y)} for scenario tiers (incl. 'hist') and idealised runs."""
        theta = np.asarray(theta, float)
        s = dict(zip(NAMES, theta))
        pam = pamset_udm(theta)
        out = {}
        for tier in tiers or self.tiers:
            r = self.scen[tier].run(pam, *self.scenario_forcing(tier, s))
            T = r[T_VAR] - r[T_VAR][self.base].mean()
            out[f"{tier}/T"] = (self.years, T)
            out[f"{tier}/N"] = (self.years, r["RIB_glob"].copy())
        if idealised:
            for exp, (e, f) in self.ideal.items():
                F = s["s_co2"] * f
                r = e.run(pam, F, F)
                yi = np.arange(1, len(F) + 1)
                out[f"{exp}/T"] = (yi, r[T_VAR].copy())
                out[f"{exp}/N"] = (yi, r["RIB_glob"].copy())
        return out

    def diagnostics(self, theta):
        """ECS-like (Gregory 1-150 on the emulated abrupt-4x), TCR, F2x, ERF terms in 2019."""
        s = dict(zip(NAMES, theta))
        o = self.run(theta, tiers=[], idealised=True)
        T4, N4 = o["abrupt-4xCO2/T"][1][:150], o["abrupt-4xCO2/N"][1][:150]
        b, a = np.polyfit(T4, N4, 1)
        T1 = o["1pctCO2/T"][1]
        t = self.tiers[0]
        i = 2019 - SF.NYSTART
        aer = lambda k: 0.5 * (self.c[f"{t}/{k}_n"][i] + self.c[f"{t}/{k}_s"][i])  # noqa: E731
        f4 = s["s_co2"] * self.c["abrupt-4xCO2/co2"][0]
        return {"ECS_gregory": -a / b / 2, "lambda_gregory": b, "F4x_gregory": a, "F4x": f4,
                "ECS_eq": s["lambda"] * f4 / 2, "TCR": T1[60:80].mean(),
                "ERF_ari_2019": s["s_ari"] * aer("ari"), "ERF_aci_2019": s["s_aci"] * aer("aci"),
                "ERF_aer_2019": s["s_ari"] * aer("ari") + s["s_aci"] * aer("aci")}


def default_theta():
    th = 0.5 * (LO + HI)
    d = dict(lambda_=0.8, rlamdo=15.0, akapa=0.66, cpi=0.21, W=2.2, beto=6.9, mixed=107.0, ocean_efficacy=1.0)
    for k, v in d.items():
        th[NAMES.index(k.rstrip("_"))] = v
    for k in ("s_co2", "s_ari", "s_aci", "s_other", "s_volc"):
        th[NAMES.index(k)] = 1.0
    th[NAMES.index("a_state")] = 0.0
    th[NAMES.index("log_sdT")] = np.log(0.1)
    th[NAMES.index("log_sdN")] = np.log(0.3)
    th[NAMES.index("rho_d")] = 0.9
    th[NAMES.index("log_sm")] = np.log(0.03)
    return th


if __name__ == "__main__":
    import time

    em = Emulator(["H", "M", "VL"])
    for a in (0.0, -0.08):
        th = default_theta()
        th[NAMES.index("a_state")] = a
        t = time.time()
        for _ in range(3):
            o = em.run(th)
        print(f"a_state={a}: {(time.time() - t) / 3:.3f} s per evaluation (3 tiers + 2 idealised)")
        print({k: round(float(v), 3) for k, v in em.diagnostics(th).items()})
        for k in ("H/T", "VL/T", "M/N"):
            x, y = o[k]
            print(" ", k, "2019", round(y[x == 2019][0], 2), "2100", round(y[-1], 2))

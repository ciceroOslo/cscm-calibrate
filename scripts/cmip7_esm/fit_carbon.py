"""Stage 2: carbon-cycle calibration on an ESM's emissions-driven CMIP7 runs.

    python scripts/cmip7_esm/fit_carbon.py fit UKESM1-3-LL [--variant eps] [--holdout H]
    python scripts/cmip7_esm/fit_carbon.py predict UKESM1-3-LL [--variant eps] [--holdout H]

Thermal and forcing parameters are taken from the stage-1 (concentration-driven) full fit
of the same variant: fitting is conditional on the stage-1 maximum-posterior set, and
predictions pair carbon-cycle posterior draws with stage-1 posterior draws.

Emissions-driven CICERO-SCM runs reproduce the stage-1 forcing scalings: aerosol
q-coefficients are scaled (s_ari, s_aci), CO2 and "other" ERF are rescaled in the
handler output each year (other uses the concentration-driven component pattern of the
same tier), volcanic is scaled, and the state-dependent feedback gets last year's GSAT.

Targets (esm_targets.load_emissions): GSAT, CO2 concentration and ocean uptake for
esm-hist (once) and each emissions-driven tier; same block likelihood as stage 1
(internal variability + persistent AR(1) discrepancy + warming/CO2-increment scaling).
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulator as EM  # noqa: E402
import esm_targets as TG  # noqa: E402
import fit_esm as F1  # noqa: E402
import scm_forcing as SF  # noqa: E402

OUT = SF.OUT / "carbon"
CARBON = [
    ("beta_f", 0.05, 1.0),          # CO2 fertilisation
    ("mixed_carbon", 25.0, 175.0),  # ocean carbon mixed-layer depth (m)
    ("solubility_sens", 0.0, 0.05),  # solubility temperature sensitivity
    ("t_half", 0.2, 1.0),           # land feedback temperature response
    ("t_threshold", 2.0, 10.0),
    ("w_threshold", 2.0, 8.0),
    ("w_sigmoid", 2.0, 8.0),
    ("ml_fracmax", 0.0, 1.0),       # mixed-layer shoaling with warming
    ("ml_w_sigmoid", 2.0, 7.0),
    ("log_sdT", -6.0, 0.0),         # discrepancy sd: GSAT (K)
    ("log_sdC", -3.0, 3.5),         # CO2 (ppm)
    ("log_sdO", -5.0, 0.5),         # ocean uptake (PgC/yr)
    ("rho_d", 0.0, 0.995),
    ("log_sm", -7.0, -0.7),         # fractional scaling error on T warming / CO2 increase
]
CNAMES = [p[0] for p in CARBON]
CLO = np.array([p[1] for p in CARBON])
CHI = np.array([p[2] for p in CARBON])
CARBON_PARAMS = CNAMES[:9]
CARBON_SET = {"solubility_limit": 0.1, "ml_t_half": 0.0}
Q_DEFAULT = {"qdirso2": -0.00308, "qindso2": -0.97 / 57.052577209999995, "qbc": 0.0279, "qoc": -0.00433}
E_TIERS = ["VL", "M", "H"]
SD_KEY = {"T": "log_sdT", "C": "log_sdC", "O": "log_sdO"}


class EExp:
    """Emissions-driven scen7 tier with stage-1 forcing scalings and state feedback."""

    def __init__(self, tier, comps):
        self.tier = tier
        sd = SF.scenario_scendata(tier)
        sd.update(conc_run=False, emstart=1850)
        self.m = SF._CSCM(sd)
        self.c = comps
        self.s = None
        self._orig = self.m.ce_handler.conc2forc
        self.m.ce_handler.conc2forc = self._conc2forc
        self.years = np.arange(SF.NYSTART, SF.NYEND + 1)
        self.base = (self.years >= 1850) & (self.years <= 1900)

    def _conc2forc(self, yr, rf_luc, rf_sun):
        tot, fn, fs, _ = self._orig(yr, rf_luc, rf_sun)
        i = yr - SF.NYSTART
        s = self.s
        co2 = self.m.ce_handler.forc["CO2"][i]
        on, os_ = self.c[f"{self.tier}/other_n"][i], self.c[f"{self.tier}/other_s"][i]
        fn = fn + (s["s_co2"] - 1) * co2 + (s["s_other"] - 1) * on
        fs = fs + (s["s_co2"] - 1) * co2 + (s["s_other"] - 1) * os_
        tot = tot + (s["s_co2"] - 1) * co2 + (s["s_other"] - 1) * 0.5 * (on + os_)
        w = self.m.results["dT_glob"][i - 1] if i > 0 else 0.0
        return tot, fn, fs, w

    def run(self, theta, carbon):
        s = dict(zip(EM.NAMES, theta))
        self.s = s
        vn, vs = self.c[f"{self.tier}/volc_n"], self.c[f"{self.tier}/volc_s"]
        self.m._volc_n_arr, self.m._volc_s_arr = s["s_volc"] * vn, s["s_volc"] * vs
        self.m._volc_n_mean, self.m._volc_s_mean = self.m._volc_n_arr.mean(axis=1), self.m._volc_s_arr.mean(axis=1)
        emi = {"qdirso2": Q_DEFAULT["qdirso2"] * s["s_ari"], "qbc": Q_DEFAULT["qbc"] * s["s_ari"],
               "qoc": Q_DEFAULT["qoc"] * s["s_ari"], "qindso2": Q_DEFAULT["qindso2"] * s["s_aci"], "qbmb": 0.0}
        pam = EM.pamset_udm(theta)
        self.m._run({"results_as_dict": True, "carbon_cycle_outputs": True},
                    pamset_udm=pam, pamset_emiconc=emi, pamset_carbon={**CARBON_SET, **carbon})
        r = self.m.results
        if pam["delta_lambda_aero"] != 0.0 and r["dynamic_lambda"].min() < 0.05:
            raise FloatingPointError("feedback parameter fell below 0.05")
        T = r[EM.T_VAR] - r[EM.T_VAR][self.base].mean()
        return {"T": T, "C": r["concentrations"]["CO2"].to_numpy(),
                "O": r["carbon cycle"]["Ocean carbon flux"].to_numpy(),
                "L": r["carbon cycle"]["Biosphere carbon flux"].to_numpy()}


_X = {}


def stage1(source, variant, driver="concentration"):
    d = F1.run_dir(source, driver, variant, None)
    meta = json.load(open(d / "meta.json"))
    return np.array(meta["map"]), np.load(d / "posterior.npy")


def _init(source, variant, holdout, tiers, theta1=None):
    comps = SF.load()
    blocks = TG.load_emissions(source)
    if holdout:
        blocks = {k: v for k, v in blocks.items() if not k.startswith(f"{holdout}/")}
    for b in blocks.values():
        x = b["x"].astype(int)
        b["D"] = np.abs(x[:, None] - x[None, :])
        b["C_iv"] = b["sigma_iv"] ** 2 * b["rho"] ** b["D"]
    _X.update(blocks=blocks, exps={t: EExp(t, comps) for t in tiers},
              theta1=stage1(source, variant)[0] if theta1 is None else theta1,
              hist=next(t for t in F1.HIST_TIER_ORDER if t in tiers))


def run_blocks(theta1, carbon):
    out = {}
    for t, e in _X["exps"].items():
        r = e.run(theta1, carbon)
        for v, y in r.items():
            out[f"{t}/{v}"] = (e.years, y)
    h = _X["hist"]
    for v in ("T", "C", "O", "L"):
        out[f"hist/{v}"] = out[f"{h}/{v}"]
    return out


def basis(key, x, y):
    block, var = key.split("/")
    if var == "O":
        return None
    if block in SF.TIERS:
        return np.where(x > F1.SPLIT_YEAR, y - np.interp(F1.SPLIT_YEAR, x, y), 0.0)
    return y - (y[0] if var == "C" else 0.0)


def log_prob(c):
    c = np.asarray(c)
    if np.any(c < CLO) or np.any(c > CHI):
        return -np.inf
    p = dict(zip(CNAMES, c))
    carbon = {k: float(p[k]) for k in CARBON_PARAMS}
    try:
        o = run_blocks(_X["theta1"], carbon)
        ll = 0.0
        for key, b in _X["blocks"].items():
            var = key.split("/")[1]
            x, y = o[key]
            ym = np.interp(b["x"], x, y)
            ll += F1.block_loglik(b, ym - b["y"], np.exp(p[SD_KEY[var]]), p["rho_d"], np.exp(p["log_sm"]),
                                  basis(key, b["x"], ym))
    except (FloatingPointError, np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return -np.inf
    return ll if np.isfinite(ll) else -np.inf


def _neg(c):
    v = log_prob(c)
    return 1e12 if not np.isfinite(v) else -v


def run_dir(source, variant, holdout):
    return OUT / f"{source}_{variant}_{'full' if not holdout else 'no' + holdout}"


def e_tiers(source, holdout):
    have = sorted({k.split("/")[0] for k in TG.load_emissions(source)} - {"hist"})
    return [t for t in have if t != holdout]


def fit(source, variant, holdout, workers, de_iter, nsteps, nwalkers, seed=1):
    import emcee
    from scipy.optimize import differential_evolution

    tiers = e_tiers(source, holdout)
    out = run_dir(source, variant, holdout)
    out.mkdir(parents=True, exist_ok=True)
    print(f"carbon {source} {variant} holdout={holdout} tiers={tiers} -> {out}", flush=True)
    with Pool(workers, initializer=_init, initargs=(source, variant, holdout, tiers)) as pool:
        t = time.time()
        de = differential_evolution(_neg, list(zip(CLO, CHI)), maxiter=de_iter, popsize=10, seed=seed,
                                    init="latinhypercube", polish=False, updating="deferred", workers=pool.map)
        print(f"DE: -logp {de.fun:.1f} after {de.nfev} evals, {time.time() - t:.0f}s", flush=True)
        pop = de.population[np.argsort(de.population_energies)][:nwalkers]
        rng = np.random.default_rng(seed)
        p0 = np.clip(pop + 1e-3 * (CHI - CLO) * rng.standard_normal(pop.shape), CLO + 1e-9, CHI - 1e-9)
        sampler = emcee.EnsembleSampler(nwalkers, len(CNAMES), log_prob, pool=pool,
                                        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        t = time.time()
        sampler.run_mcmc(p0, nsteps, progress=False)
        print(f"MCMC: {nsteps} x {nwalkers}, {time.time() - t:.0f}s, acceptance "
              f"{sampler.acceptance_fraction.mean():.2f}", flush=True)
    chain, lp = sampler.get_chain(), sampler.get_log_prob()
    burn = nsteps // 2
    flat = chain[burn:].reshape(-1, len(CNAMES))
    np.save(out / "posterior.npy", flat)
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:  # noqa: BLE001
        tau = np.full(len(CNAMES), np.nan)
    json.dump({"source": source, "variant": variant, "holdout": holdout, "tiers": tiers, "params": CNAMES,
               "acceptance": float(sampler.acceptance_fraction.mean()), "tau": dict(zip(CNAMES, map(float, tau))),
               "map": flat[np.argmax(lp[burn:].ravel())].tolist(), "de_mode": de.x.tolist()},
              open(out / "meta.json", "w"), indent=1)
    q = np.percentile(flat, [5, 50, 95], axis=0)
    frac = (q - CLO[None]) / (CHI - CLO)[None]
    tab = pd.DataFrame(q.T, index=CNAMES, columns=["p05", "p50", "p95"]).assign(tau=tau)
    tab["at_bound"] = np.where(frac[0] < 0.02, "lower", np.where(frac[2] > 0.98, "upper", ""))
    print(tab.round(3), flush=True)


# ------------------------------------------------------------------ prediction
def _pinit(source, variant):
    _init(source, variant, None, SF.TIERS)


def _prun(args):
    theta1, c = args
    p = dict(zip(CNAMES, c))
    o = run_blocks(theta1, {k: float(p[k]) for k in CARBON_PARAMS})
    return {k: v[1].astype(np.float64) for k, v in o.items()}


def predict(source, variant, holdout, workers, ndraw=200, seed=3):
    out = run_dir(source, variant, holdout)
    meta = json.load(open(out / "meta.json"))
    post = np.load(out / "posterior.npy")
    th_post = stage1(source, variant)[1]
    rng = np.random.default_rng(seed)
    cd = post[rng.choice(len(post), ndraw, replace=False)]
    td = th_post[rng.choice(len(th_post), ndraw, replace=False)]
    with Pool(workers, initializer=_pinit, initargs=(source, variant)) as pool:
        res = pool.map(_prun, list(zip(td, cd)))
    sims = {k: np.stack([r[k] for r in res]) for k in res[0]}
    years = np.arange(SF.NYSTART, SF.NYEND + 1)
    p = dict(zip(CNAMES, cd.T))
    sm = np.exp(p["log_sm"])
    rows, total = [], {}
    for key, arr in sims.items():
        block, var = key.split("/")
        if var == "L":
            tot = arr
        else:
            sd = np.exp(p[SD_KEY[var]])
            tot = arr + np.stack([F1.ar1(rng, 1, arr.shape[1], sd[i], p["rho_d"][i])[0] for i in range(len(arr))])
            u = [basis(key, years, a) for a in arr]
            if u[0] is not None:
                tot = tot + (sm * rng.standard_normal(len(arr)))[:, None] * np.stack(u)
        total[key] = tot
        pc = np.percentile(arr, [5, 50, 95], axis=0)
        pt = np.percentile(tot, [5, 95], axis=0)
        for i, xx in enumerate(years):
            rows.append((block, var, int(xx), pc[0, i], pc[1, i], pc[2, i], pt[0, i], pt[1, i]))
    pr = pd.DataFrame(rows, columns=["block", "var", "x", "p05", "p50", "p95", "p05_tot", "p95_tot"])
    pr.to_csv(out / "predictions.csv.gz", index=False, float_format="%.4f")
    blocks = TG.load_emissions(source)
    sc = []
    i0 = 2081 - SF.NYSTART
    for key, b in blocks.items():
        block, var = key.split("/")
        idx = np.searchsorted(years, b["x"])
        med = np.median(sims[key][:, idx], axis=0)
        noisy = total[key][:, idx] + b["sigma_iv"] * rng.standard_normal((ndraw, len(idx)))
        lo, hi = np.percentile(noisy, [5, 95], axis=0)
        e = {"block": key, "held_out": block == holdout, "rmse": float(np.sqrt(np.mean((med - b["y"]) ** 2))),
             "bias": float(np.mean(med - b["y"])), "frac_in_90": float(((b["y"] >= lo) & (b["y"] <= hi)).mean())}
        if block in SF.TIERS:
            w = b["x"] >= 2081
            esm = b["y"][w].mean()
            tot = total[key][:, i0:].mean(axis=1)
            e.update(esm_2081_2100=float(esm), emu_p50=float(np.median(sims[key][:, i0:].mean(axis=1))),
                     tot_p05=float(np.percentile(tot, 5)), tot_p95=float(np.percentile(tot, 95)),
                     pit_tot=float((tot < esm).mean()))
        sc.append(e)
    sc = pd.DataFrame(sc)
    sc.to_csv(out / "scores.csv", index=False)
    print(f"carbon {source} {variant} holdout={holdout}: acceptance {meta['acceptance']:.2f}")
    print(sc.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "predict"])
    ap.add_argument("source")
    ap.add_argument("--variant", default="eps", choices=list(EM.VARIANTS))
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--de-iter", type=int, default=40)
    ap.add_argument("--nsteps", type=int, default=600)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--ndraw", type=int, default=200)
    a = ap.parse_args()
    if a.stage == "fit":
        fit(a.source, a.variant, a.holdout, a.workers, a.de_iter, a.nsteps, a.nwalkers)
    else:
        predict(a.source, a.variant, a.holdout, a.workers, a.ndraw)

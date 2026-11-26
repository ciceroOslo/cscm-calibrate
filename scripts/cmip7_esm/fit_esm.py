"""Joint per-ESM calibration of CICERO-SCM on all of an ESM's CMIP7 output.

    python scripts/cmip7_esm/fit_esm.py fit CanESM6-0-MR [--variant eps|eps_state] [--holdout H]
    python scripts/cmip7_esm/fit_esm.py predict CanESM6-0-MR [--variant ...] [--holdout H]

Blocks: history once (1850-2021), each scenario tier (2022-2100), abrupt-4xCO2 and
1pctCO2; GSAT (T) and net TOA flux (N). Each block has a Gaussian likelihood with full
covariance

    Sigma = sigma_iv^2 R(rho_iv)/k  +  sd^2 R(rho_d)  +  sm^2 u u^T   (last term T only)

  internal variability   sigma_iv, rho_iv from piControl / member spread (esm_targets.py)
  persistent discrepancy AR(1), sd (per variable) and rho_d sampled
  scaling discrepancy    a block-specific fractional error on the model's warming u
                         (scenario tiers: warming since 2021; history and idealised: T),
                         sm sampled; marginalised analytically (rank-one term)

A held-out tier is left out of the likelihood and predicted. Predictions add draws of
both discrepancy terms, so the "total" range is the emulator's predictive range for a
tier the ESM has not run.

fit:      differential evolution for the mode, then emcee from the final DE population.
predict:  run thinned posterior draws over all seven tiers + idealised runs; write
          percentiles, scores against every ESM block and derived metrics.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulator as EM  # noqa: E402
import esm_targets as TG  # noqa: E402
import scm_forcing as SF  # noqa: E402

OUT = SF.OUT / "fits"
HIST_TIER_ORDER = ["M", "H", "VL", "ML", "L", "HL", "LN"]
SPLIT_YEAR = 2021  # last shared historical year
_EM = _BLOCKS = _HIST = _SPACE = None


def run_dir(source, driver, variant, holdout):
    return OUT / f"{source}_{driver[:4]}_{variant}_{'full' if not holdout else 'no' + holdout}"


def _fit_tiers(source, driver, holdout):
    tiers = sorted({k.split("/")[0] for k in TG.load(source, driver)} - {"hist", "abrupt-4xCO2", "1pctCO2"})
    return [t for t in tiers if t != holdout]


def prepare_blocks(source, driver, holdout=None):
    blocks = TG.load(source, driver)
    if holdout:
        blocks = {k: v for k, v in blocks.items() if not k.startswith(f"{holdout}/")}
    for b in blocks.values():
        x = b["x"].astype(int)
        b["D"] = np.abs(x[:, None] - x[None, :])
        b["C_iv"] = b["sigma_iv"] ** 2 * b["rho"] ** b["D"]
    return blocks


def _init(source, driver, variant, holdout, tiers):
    global _EM, _BLOCKS, _HIST, _SPACE
    _SPACE = EM.Space(variant)
    _BLOCKS = prepare_blocks(source, driver, holdout)
    _EM = EM.Emulator(tiers)
    _HIST = next(t for t in HIST_TIER_ORDER if t in tiers)


def model_blocks(theta, em=None, hist_tier=None, tiers=None):
    em = em or _EM
    o = em.run(theta, tiers=tiers)
    ht = hist_tier or _HIST
    o["hist/T"], o["hist/N"] = o[f"{ht}/T"], o[f"{ht}/N"]
    return o


def scaling_basis(key, x, y):
    """Warming the fractional scaling error acts on: since 2021 for scenario tiers, else T."""
    block = key.split("/")[0]
    if block in SF.TIERS:
        return np.where(x > SPLIT_YEAR, y - np.interp(SPLIT_YEAR, x, y), 0.0)
    return y


def block_loglik(b, r, sd, rho_d, sm=0.0, u=None):
    cov = b["C_iv"] + sd**2 * rho_d ** b["D"]
    if u is not None and sm > 0:
        cov = cov + sm**2 * np.outer(u, u)
    c = cho_factor(cov, lower=True, check_finite=False)
    return -0.5 * r @ cho_solve(c, r, check_finite=False) - np.log(np.diag(c[0])).sum()


def log_prob(free):
    free = np.asarray(free)
    if np.any(free < _SPACE.lo) or np.any(free > _SPACE.hi):
        return -np.inf
    theta = _SPACE.full(free)
    s = dict(zip(EM.NAMES, theta))
    sd = {"T": np.exp(s["log_sdT"]), "N": np.exp(s["log_sdN"])}
    sm = np.exp(s["log_sm"])
    try:
        o = model_blocks(theta)
        ll = 0.0
        for key, b in _BLOCKS.items():
            var = key.split("/")[1]
            x, y = o[key]
            ym = np.interp(b["x"], x, y)
            u = scaling_basis(key, b["x"], ym) if var == "T" else None
            ll += block_loglik(b, ym - b["y"], sd[var], s["rho_d"], sm, u)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return -np.inf
    return ll if np.isfinite(ll) else -np.inf


def _neg(free):
    v = log_prob(free)
    return 1e12 if not np.isfinite(v) else -v


def fit(source, driver, variant, holdout, workers, de_iter, nsteps, nwalkers, seed=1):
    import emcee
    from scipy.optimize import differential_evolution

    space = EM.Space(variant)
    tiers = _fit_tiers(source, driver, holdout)
    out = run_dir(source, driver, variant, holdout)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{source} {driver} {variant} holdout={holdout} tiers={tiers} -> {out}", flush=True)
    with Pool(workers, initializer=_init, initargs=(source, driver, variant, holdout, tiers)) as pool:
        t = time.time()
        de = differential_evolution(_neg, list(zip(space.lo, space.hi)), maxiter=de_iter, popsize=10, seed=seed,
                                    init="latinhypercube", polish=False, updating="deferred",
                                    workers=pool.map, tol=1e-6)
        print(f"DE: -logp {de.fun:.1f} after {de.nfev} evals, {time.time() - t:.0f}s", flush=True)
        print("  mode:", {k: round(float(v), 3) for k, v in zip(space.names, de.x)}, flush=True)
        pop = de.population[np.argsort(de.population_energies)]
        p0 = pop[:nwalkers].copy()
        rng = np.random.default_rng(seed)
        p0 += 1e-3 * (space.hi - space.lo) * rng.standard_normal(p0.shape)
        p0 = np.clip(p0, space.lo + 1e-9, space.hi - 1e-9)
        sampler = emcee.EnsembleSampler(nwalkers, len(space.names), log_prob, pool=pool,
                                        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        t = time.time()
        sampler.run_mcmc(p0, nsteps, progress=False)
        print(f"MCMC: {nsteps} steps x {nwalkers} walkers, {time.time() - t:.0f}s, "
              f"acceptance {sampler.acceptance_fraction.mean():.2f}", flush=True)
    chain, lp = sampler.get_chain(), sampler.get_log_prob()
    np.save(out / "chain.npy", chain)
    np.save(out / "log_prob.npy", lp)
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:  # noqa: BLE001
        tau = np.full(len(space.names), np.nan)
    burn = nsteps // 2
    flat = chain[burn:].reshape(-1, len(space.names))
    full = np.array([space.full(f) for f in flat])
    np.save(out / "posterior.npy", full)  # full theta (fixed parameters filled in)
    meta = {"source": source, "driver": driver, "variant": variant, "holdout": holdout, "tiers": tiers,
            "params": EM.NAMES, "free": space.names, "fixed": space.fixed,
            "bounds": {n: [float(lo), float(hi)] for n, lo, hi in zip(EM.NAMES, EM.LO, EM.HI)},
            "de_mode": space.full(de.x).tolist(), "de_logp": -float(de.fun), "nsteps": nsteps,
            "nwalkers": nwalkers, "burn": burn, "acceptance": float(sampler.acceptance_fraction.mean()),
            "tau": dict(zip(space.names, map(float, tau))),
            "map": full[np.argmax(lp[burn:].ravel())].tolist(), "max_logp": float(lp[burn:].max())}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    q = np.percentile(flat, [5, 50, 95], axis=0)
    rng_frac = (q - space.lo[None]) / (space.hi - space.lo)[None]
    tab = pd.DataFrame(q.T, index=space.names, columns=["p05", "p50", "p95"]).assign(tau=tau)
    tab["at_bound"] = np.where(rng_frac[0] < 0.02, "lower", np.where(rng_frac[2] > 0.98, "upper", ""))
    print(tab.round(3), flush=True)


# ------------------------------------------------------------------ prediction
_PEM = None


def _pinit():
    global _PEM
    _PEM = EM.Emulator(SF.TIERS)


def _prun(theta):
    o = model_blocks(theta, em=_PEM, hist_tier="M", tiers=SF.TIERS)
    return {k: v[1].astype(np.float64) for k, v in o.items()}, _PEM.diagnostics(theta)


def ar1(rng, n_draw, n_t, sd, rho):
    e = np.empty((n_draw, n_t))
    e[:, 0] = sd * rng.standard_normal(n_draw)
    k = np.sqrt(1 - rho**2)
    for t in range(1, n_t):
        e[:, t] = rho * e[:, t - 1] + k * sd * rng.standard_normal(n_draw)
    return e


def predict(source, driver, variant, holdout, workers, ndraw=300, seed=2):
    out = run_dir(source, driver, variant, holdout)
    meta = json.load(open(out / "meta.json"))
    post = np.load(out / "posterior.npy")
    rng = np.random.default_rng(seed)
    draws = post[rng.choice(len(post), size=min(ndraw, len(post)), replace=False)]
    with Pool(workers, initializer=_pinit) as pool:
        res = [r for r in pool.map(_prun, list(draws))]
    sims = {k: np.stack([r[0][k] for r in res]) for k in res[0][0]}
    diag = pd.DataFrame([r[1] for r in res])
    diag.to_csv(out / "diagnostics.csv", index=False)
    years = np.arange(SF.NYSTART, SF.NYEND + 1)
    s = dict(zip(EM.NAMES, draws.T))
    sm, rho_d = np.exp(s["log_sm"]), s["rho_d"]
    total, rows = {}, []
    for key, arr in sims.items():
        block, var = key.split("/")
        x = years if block in SF.TIERS or block == "hist" else np.arange(1, arr.shape[1] + 1)
        sd = np.exp(s["log_sdT" if var == "T" else "log_sdN"])
        disc = np.stack([ar1(rng, 1, arr.shape[1], sd[i], rho_d[i])[0] for i in range(len(arr))])
        tot = arr + disc
        if var == "T":
            u = np.stack([scaling_basis(key, x, a) for a in arr])
            tot = tot + (sm * rng.standard_normal(len(arr)))[:, None] * u
        total[key] = tot
        pc = np.percentile(arr, [5, 50, 95], axis=0)
        pt = np.percentile(tot, [5, 95], axis=0)
        for i, xx in enumerate(x):
            rows.append((block, var, int(xx), pc[0, i], pc[1, i], pc[2, i], pt[0, i], pt[1, i]))
    pr = pd.DataFrame(rows, columns=["block", "var", "x", "p05", "p50", "p95", "p05_tot", "p95_tot"])
    pr.to_csv(out / "predictions.csv.gz", index=False, float_format="%.4f")
    i0 = 2081 - SF.NYSTART
    np.savez(out / "T_2081_2100.npz", **{t: total[f"{t}/T"][:, i0:].mean(axis=1) for t in SF.TIERS},
             **{f"{t}_param": sims[f"{t}/T"][:, i0:].mean(axis=1) for t in SF.TIERS})
    # scores against every available ESM block (fitted or held out)
    blocks = TG.load(source, driver)
    sc = []
    for key, b in blocks.items():
        block, var = key.split("/")
        xx = years if block in SF.TIERS or block == "hist" else np.arange(1, sims[key].shape[1] + 1)
        idx = np.searchsorted(xx, b["x"])
        med = np.median(sims[key][:, idx], axis=0)
        noisy = total[key][:, idx] + b["sigma_iv"] * rng.standard_normal((len(draws), len(idx)))
        lo, hi = np.percentile(noisy, [5, 95], axis=0)
        r = med - b["y"]
        entry = {"block": key, "held_out": block == holdout, "rmse": float(np.sqrt(np.mean(r**2))),
                 "bias": float(np.mean(r)), "frac_in_90": float(((b["y"] >= lo) & (b["y"] <= hi)).mean())}
        if block in SF.TIERS and var == "T":
            w = (b["x"] >= 2081) & (b["x"] <= 2100)
            esm = b["y"][w].mean()
            par = sims[key][:, i0:].mean(axis=1)
            neff = w.sum() * (1 - b["rho"]) / (1 + b["rho"])
            tot = total[key][:, i0:].mean(axis=1) + b["sigma_iv"] / np.sqrt(neff) * rng.standard_normal(len(draws))
            entry.update(esm_2081_2100=float(esm), emu_p50=float(np.median(par)),
                         par_p05=float(np.percentile(par, 5)), par_p95=float(np.percentile(par, 95)),
                         tot_p05=float(np.percentile(tot, 5)), tot_p95=float(np.percentile(tot, 95)),
                         pit_par=float((par < esm).mean()), pit_tot=float((tot < esm).mean()))
        sc.append(entry)
    sc = pd.DataFrame(sc)
    sc.to_csv(out / "scores.csv", index=False)
    print(f"{source} {driver} {variant} holdout={holdout}: acceptance {meta['acceptance']:.2f}")
    print(sc.round(3).to_string(index=False))
    m = TG.metrics(source)
    d = diag.describe(percentiles=[0.05, 0.5, 0.95]).T[["5%", "50%", "95%"]]
    d["esm"] = [m.get("ECS") if c.startswith("ECS") else m.get("TCR") if c == "TCR" else np.nan for c in d.index]
    print(d.round(3).to_string())
    print("discrepancy: sdT {:.3f} K, sdN {:.3f} W/m2, rho_d {:.3f}, sm {:.3f}".format(
        *np.median(np.c_[np.exp(s["log_sdT"]), np.exp(s["log_sdN"]), rho_d, sm], axis=0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "predict"])
    ap.add_argument("source")
    ap.add_argument("--driver", default="concentration")
    ap.add_argument("--variant", default="eps", choices=list(EM.VARIANTS))
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--de-iter", type=int, default=60)
    ap.add_argument("--nsteps", type=int, default=800)
    ap.add_argument("--nwalkers", type=int, default=40)
    ap.add_argument("--ndraw", type=int, default=300)
    a = ap.parse_args()
    if a.stage == "fit":
        fit(a.source, a.driver, a.variant, a.holdout, a.workers, a.de_iter, a.nsteps, a.nwalkers)
    else:
        predict(a.source, a.driver, a.variant, a.holdout, a.workers, a.ndraw)

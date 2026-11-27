"""
Fit the emulator to one ESM: likelihood, sampler and held-out tiers

The likelihood and sampler are described in
``docs/further-background/esm-emulation.md`` (section Fit).

    python -m cscm_calibrate.esm.fit <data_dir> <forcing.npz> <source> <out_dir>
        [--holdout TIER] [--workers 24]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import emcee
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution

from cscm_calibrate.esm.emulator import PARAMS, Emulator
from cscm_calibrate.esm.forcing import TIERS
from cscm_calibrate.esm.targets import HIST_END, idealised_blocks, scenario_blocks

DISCREPANCY = {
    "log_sdT": (-6.0, 0.0),  # log persistent discrepancy sd, GSAT (K)
    "log_sdN": (-6.0, 0.5),  # log persistent discrepancy sd, TOA flux (W m-2)
    "rho_d": (0.0, 0.995),  # discrepancy lag-1 autocorrelation
    "log_sm": (-7.0, -0.7),  # log fractional warming error per block
}
BOUNDS = {**PARAMS, **DISCREPANCY}
NAMES = list(BOUNDS)
LO, HI = np.array(list(BOUNDS.values())).T
# tier whose run provides the history block (they share forcing to 2021)
HIST_TIER_ORDER = ["M", "H", "VL", "ML", "L", "HL", "LN"]
END_CENTURY = (2081, 2100)  # period scored for a held-out tier


def scaling_basis(key: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Warming the scaling error acts on: since 2021 for a tier, else all of it."""
    if key.split("/")[0] in TIERS:
        return np.where(x > HIST_END, y - np.interp(HIST_END, x, y), 0.0)
    return y


def block_loglik(b, r, sd, rho_d, sm=0.0, u=None) -> float:  # noqa: PLR0913
    """Gaussian log-likelihood of residual ``r`` for target block ``b``."""
    lag = np.abs(np.subtract.outer(b.x, b.x))
    cov = b.sigma**2 * b.rho**lag + sd**2 * rho_d**lag
    if u is not None:
        cov = cov + sm**2 * np.outer(u, u)
    c = cho_factor(cov, lower=True, check_finite=False)
    logdet = 2 * np.log(np.diag(c[0])).sum()
    return -0.5 * (r @ cho_solve(c, r) + logdet + len(r) * np.log(2 * np.pi))


class LogPosterior:
    """Log posterior of a parameter vector (ordered as ``NAMES``)."""

    def __init__(self, blocks: dict, comps: dict, tiers: list[str]):
        self.blocks = blocks
        self.em = Emulator(comps, tiers)
        self.hist = next(t for t in HIST_TIER_ORDER if t in tiers)

    def __call__(self, x) -> float:
        """Evaluate at ``x``; -inf outside the prior bounds or if a run fails."""
        if np.any(x < LO) or np.any(x > HI):
            return -np.inf
        p = dict(zip(NAMES, x))
        sd = {"T": np.exp(p["log_sdT"]), "N": np.exp(p["log_sdN"])}
        ll = 0.0
        try:
            o = self.em.run(p)
            o["hist/T"], o["hist/N"] = o[f"{self.hist}/T"], o[f"{self.hist}/N"]
            for key, b in self.blocks.items():
                var = key.split("/")[1]
                y = np.interp(b.x, *o[key])
                u = scaling_basis(key, b.x, y) if var == "T" else None
                ll += block_loglik(
                    b, y - b.y, sd[var], p["rho_d"], np.exp(p["log_sm"]), u
                )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf


# one LogPosterior per worker process (the emulator holds unpicklable state)
_LOGP = None


def _init(*args):
    global _LOGP  # noqa: PLW0603
    _LOGP = LogPosterior(*args)


def _logp(x):
    return _LOGP(x)


def _cost(x):
    v = _LOGP(x)
    return -v if np.isfinite(v) else 1e12


def fit(  # noqa: PLR0913
    blocks, comps, tiers, workers=24, de_iter=60, nsteps=800, nwalkers=40, seed=1
) -> dict:
    """Sample the posterior; returns the post-burn-in samples and diagnostics."""
    with Pool(workers, initializer=_init, initargs=(blocks, comps, tiers)) as pool:
        de = differential_evolution(
            _cost,
            list(zip(LO, HI)),
            maxiter=de_iter,
            popsize=10,
            seed=seed,
            init="latinhypercube",
            polish=False,
            updating="deferred",
            workers=pool.map,
            tol=1e-6,
        )
        # start the walkers from the best of the final DE population
        p0 = de.population[np.argsort(de.population_energies)[:nwalkers]]
        rng = np.random.default_rng(seed)
        p0 = p0 + 1e-3 * (HI - LO) * rng.standard_normal(p0.shape)
        p0 = np.clip(p0, LO + 1e-9, HI - 1e-9)
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        sampler = emcee.EnsembleSampler(
            nwalkers, len(NAMES), _logp, pool=pool, moves=moves
        )
        sampler.run_mcmc(p0, nsteps)
    burn = nsteps // 2
    return {
        "samples": sampler.get_chain(discard=burn, flat=True),
        "log_prob": sampler.get_log_prob(discard=burn, flat=True),
        "de_mode": de.x,
        "acceptance": float(sampler.acceptance_fraction.mean()),
        "tau": sampler.get_autocorr_time(tol=0),
    }


def _ar1(rng, n, sd, rho):
    e = np.empty(n)
    e[0] = sd * rng.standard_normal()
    for t in range(1, n):
        e[t] = rho * e[t - 1] + np.sqrt(1 - rho**2) * sd * rng.standard_normal()
    return e


def predict(samples, comps, tiers=TIERS, ndraw=300, seed=2) -> dict:
    """
    Emulate posterior draws with and without the discrepancy terms

    Returns ``{"<block>/<var>": (x, emulated, total)}``, where ``total`` adds
    draws of both discrepancy terms: the predictive range for an ESM run.
    """
    rng = np.random.default_rng(seed)
    draws = samples[rng.choice(len(samples), min(ndraw, len(samples)), False)]
    em = Emulator(comps, tiers)
    out = {}
    for x in draws:
        p = dict(zip(NAMES, x))
        for key, (xx, y) in em.run(p).items():
            sd = np.exp(p[f"log_sd{key[-1]}"])
            tot = y + _ar1(rng, len(y), sd, p["rho_d"])
            if key.endswith("/T"):
                u = scaling_basis(key, xx, y)
                tot = tot + np.exp(p["log_sm"]) * rng.standard_normal() * u
            out.setdefault(key, (xx, [], []))
            out[key][1].append(y)
            out[key][2].append(tot)
    return {k: (x, np.array(y), np.array(t)) for k, (x, y, t) in out.items()}


def main():
    """Fit one ESM from the command line (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for a in ("data_dir", "forcing", "source", "out_dir"):
        ap.add_argument(a)
    ap.add_argument("--holdout")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    comps = dict(np.load(a.forcing))
    blocks = scenario_blocks(a.data_dir, a.source, "concentration")
    blocks |= idealised_blocks(a.data_dir, a.source)
    held = {k: b for k, b in blocks.items() if k.startswith(f"{a.holdout}/")}
    blocks = {k: b for k, b in blocks.items() if k not in held}
    tiers = sorted({k.split("/")[0] for k in blocks} & set(TIERS))
    r = fit(blocks, comps, tiers, workers=a.workers)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "posterior.npz", names=NAMES, **r)
    meta = {"source": a.source, "holdout": a.holdout, "tiers": tiers}
    meta |= {
        "acceptance": r["acceptance"],
        "median": dict(zip(NAMES, np.median(r["samples"], 0).tolist())),
    }
    if held:
        # 2081-2100 mean GSAT of the held-out tier against the predictive range
        b = held[f"{a.holdout}/T"]
        x, _, tot = predict(r["samples"], comps, [a.holdout])[f"{a.holdout}/T"]
        lo, hi = END_CENTURY
        esm = b.y[(b.x >= lo) & (b.x <= hi)].mean()
        # add the ESM's own internal variability in a 20-year mean
        n_eff = (hi - lo + 1) * (1 - b.rho) / (1 + b.rho)
        noise = (
            b.sigma
            / np.sqrt(n_eff)
            * np.random.default_rng(3).standard_normal(len(tot))
        )
        pred = tot[:, (x >= lo) & (x <= hi)].mean(axis=1) + noise
        meta["holdout_T_2081_2100"] = {
            "esm": esm,
            "p05_p50_p95": np.percentile(pred, [5, 50, 95]).tolist(),
            "pit": float((pred < esm).mean()),
        }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()

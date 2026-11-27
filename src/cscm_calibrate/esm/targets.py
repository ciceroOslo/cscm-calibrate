"""
Fitting targets for one ESM from the ScenarioMIP ESM data products

Reads the global-mean CSV products of the scenariomip-esm-figures repository
(``data/``) and returns one :class:`Block` per time series to be fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HIST_END = 2021  # last year of the historical runs; scenarios start in 2022
N_ABRUPT = 150  # abrupt-4xCO2 years fitted
MIN_MEMBERS = 3  # members needed to estimate noise from the ensemble spread
MIN_CONTROL_YEARS = 100  # piControl years needed to estimate noise from the control

# variable: (product, column, noise floor for one member)
EMISSIONS_DRIVEN = {
    "T": ("gsat.csv", "gsat_raw_K", 0.05),
    "C": ("atmospheric_co2.csv", "co2_ppm", 0.2),
    "O": ("carbon_budget.csv", "ocean_PgC_yr", 0.05),
}
CONCENTRATION_DRIVEN = {
    "T": ("gsat.csv", "gsat_raw_K", 0.05),
    "N": ("toa_imbalance.csv", "N", 0.1),
}


@dataclass
class Block:
    """
    One ESM time series and its internal-variability noise

    ``sigma`` is for the ensemble mean (single-member sigma / sqrt(k)) and
    ``rho`` is the lag-1 autocorrelation of single-member noise.
    """

    x: np.ndarray
    y: np.ndarray
    k: int
    sigma: float
    rho: float


def _read(data_dir: Path, name: str, source: str, driver: str | None) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / name)
    df = df[df.source == source]
    if "ensemble" in df:
        df = df[df.ensemble == "ESM"]
    if driver is not None:
        df = df[df.driver == driver]
    if name == "toa_imbalance.csv":
        df = df.assign(N=df.net_toa_Wm2 - df.net_toa_control_Wm2)
    return df


def _spread_noise(df: pd.DataFrame, col: str) -> tuple[float, float] | None:
    """Single-member sigma and lag-1 rho from deviations about the ensemble mean."""
    dev = []
    for _, x in df.groupby("tier"):
        p = x.pivot_table(index="year", columns="variant", values=col).dropna(axis=1)
        k = p.shape[1]
        if k >= MIN_MEMBERS:
            dev.append(p.sub(p.mean(axis=1), axis=0).to_numpy() * np.sqrt(k / (k - 1)))
    if not dev:
        return None
    d = np.concatenate(dev, axis=1)
    return float(d.std()), float(np.corrcoef(d[:-1].ravel(), d[1:].ravel())[0, 1])


def _control_noise(data_dir: Path, source: str, var: str) -> tuple[float, float] | None:
    """Sigma and lag-1 rho of the linearly detrended piControl series."""
    col = {"T": "tas_K", "N": "N_Wm2"}.get(var)
    if col is None:
        return None
    e = _read(data_dir, "energy_closure.csv", source, None)
    c = e[e.experiment == "piControl"].sort_values("year").dropna(subset=[col])
    if len(c) < MIN_CONTROL_YEARS:
        return None
    y = c[col].to_numpy(float)
    t = np.arange(len(y))
    r = y - np.polyval(np.polyfit(t, y, 1), t)
    return float(r.std()), float(np.corrcoef(r[:-1], r[1:])[0, 1])


def scenario_blocks(data_dir: Path, source: str, driver: str) -> dict[str, Block]:
    """
    History (counted once) and each scenario tier, for every variable of a driver

    Keys are ``"<tier>/<var>"`` with tier ``"hist"`` for 1850-2021.
    """
    variables = EMISSIONS_DRIVEN if driver == "emissions" else CONCENTRATION_DRIVEN
    blocks = {}
    for var, (name, col, floor) in variables.items():
        df = _read(data_dir, name, source, driver).dropna(subset=[col])
        if df.empty:  # variable not available for this ESM
            continue
        df = df.drop_duplicates(["tier", "variant", "year"])
        fut = df[(df.year > HIST_END) & (df.tier != "hist")]
        sigma, rho = (
            _control_noise(data_dir, source, var)
            or _spread_noise(fut, col)
            or (floor, 0.5)
        )
        sigma = max(sigma, floor)
        hist = df[df.year <= HIST_END].drop_duplicates(["variant", "year"])
        for tier, x in [("hist", hist), *fut.groupby("tier")]:
            s = x.groupby("year")[col].agg(["mean", "count"])
            k = int(s["count"].median())
            blocks[f"{tier}/{var}"] = Block(
                s.index.to_numpy(), s["mean"].to_numpy(), k, sigma / np.sqrt(k), rho
            )
    return blocks


def idealised_blocks(data_dir: Path, source: str) -> dict[str, Block]:
    """abrupt-4xCO2 T and N (years 1-150) and 1pctCO2 T, relative to piControl."""
    noise = {v: _control_noise(data_dir, source, v) or (0.15, 0.5) for v in "TN"}
    blocks = {}
    g = _read(data_dir, "gregory.csv", source, None)
    g = g[g.year_index <= N_ABRUPT].dropna(subset=["dT_K", "dN_Wm2"])
    for var, col in (("T", "dT_K"), ("N", "dN_Wm2")):
        if not g.empty:
            blocks[f"abrupt-4xCO2/{var}"] = Block(
                g.year_index.to_numpy(), g[col].to_numpy(), 1, *noise[var]
            )
    t = _read(data_dir, "transient_1pct.csv", source, None).dropna(subset=["dT_K"])
    if not t.empty:
        blocks["1pctCO2/T"] = Block(
            t.year_index.to_numpy(), t.dT_K.to_numpy(), 1, *noise["T"]
        )
    return blocks

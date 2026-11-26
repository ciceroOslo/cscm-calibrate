"""CMIP7 ESM fitting targets from the scenariomip-esm-figures data products.

For one ESM and driver, return blocks of (time axis, ESM mean, members k), each with an
internal-variability noise model (sigma, lag-1 rho) from the model's piControl:

  hist/T, hist/N           1850-2021 (shared history, counted once)
  <tier>/T, <tier>/N       2022-2100 for each available scenario tier
  abrupt-4xCO2/T, /N       years 1..N4X vs piControl (gregory.csv)
  1pctCO2/T                years 1..140 vs piControl (transient_1pct.csv)

T is GSAT anomaly vs 1850-1900 (scenario blocks) or vs piControl (idealised);
N is the net TOA flux minus the model's piControl mean.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/storage/no-backup-nac/users/bensan/scenariomip-esm-figures/data")
HIST_END = 2021
N4X = 150


def _read(name):
    return pd.read_csv(DATA / name)


def control_noise(source):
    """sigma and lag-1 autocorrelation of detrended annual piControl T and N."""
    e = _read("energy_closure.csv")
    c = e[(e.source == source) & (e.experiment == "piControl")].sort_values("year")
    if c.empty:
        c = e[(e.source == source) & (e.experiment == "esm-piControl")].sort_values("year")
    out = {}
    for var, col in (("T", "tas_K"), ("N", "N_Wm2")):
        y = c[col].to_numpy(float)
        t = np.arange(len(y))
        r = y - np.polyval(np.polyfit(t, y, 1), t)
        out[var] = (float(r.std()), float(np.corrcoef(r[:-1], r[1:])[0, 1]))
    return out


def member_noise(source, driver="concentration"):
    """Single-member sigma and lag-1 rho from deviations about the ensemble mean (k>=3 tiers)."""
    out = {}
    g = _read("gsat.csv")
    toa = _read("toa_imbalance.csv")
    toa["N"] = toa.net_toa_Wm2 - toa.net_toa_control_Wm2
    for var, df, col in (("T", g, "gsat_raw_K"), ("N", toa, "N")):
        df = df[(df.ensemble == "ESM") & (df.source == source) & (df.driver == driver) & (df.year > HIST_END)]
        dev, lag = [], []
        for _, x in df.groupby("tier"):
            p = x.pivot_table(index="year", columns="variant", values=col).dropna(axis=1)
            k = p.shape[1]
            if k < 3:
                continue
            d = p.sub(p.mean(axis=1), axis=0).to_numpy() * np.sqrt(k / (k - 1))
            dev.append(d.ravel())
            lag.append(np.corrcoef(d[:-1].ravel(), d[1:].ravel())[0, 1])
        if dev:
            out[var] = (float(np.concatenate(dev).std()), float(np.mean(lag)))
    return out


def _scenario_blocks(source, driver, tiers=None):
    g = _read("gsat.csv")
    g = g[(g.ensemble == "ESM") & (g.source == source) & (g.driver == driver)]
    toa = _read("toa_imbalance.csv")
    toa = toa[(toa.ensemble == "ESM") & (toa.source == source) & (toa.driver == driver)].copy()
    toa["N"] = toa.net_toa_Wm2 - toa.net_toa_control_Wm2
    blocks = {}
    for var, df, col in (("T", g, "gsat_raw_K"), ("N", toa, "N")):
        if df.empty or df[col].isna().all():
            continue
        h = df[df.year <= HIST_END].drop_duplicates(["variant", "year"])
        h = h.groupby("year")[col].agg(["mean", "count"])
        blocks[f"hist/{var}"] = (h.index.to_numpy(), h["mean"].to_numpy(), int(h["count"].median()))
        for tier, x in df[df.year > HIST_END].groupby("tier"):
            if tiers and tier not in tiers:
                continue
            s = x.groupby("year")[col].agg(["mean", "count"]).dropna()
            blocks[f"{tier}/{var}"] = (s.index.to_numpy(), s["mean"].to_numpy(), int(s["count"].median()))
    return blocks


def _idealised_blocks(source, n4x=N4X):
    blocks = {}
    gr = _read("gregory.csv")
    gr = gr[gr.source == source].sort_values("year_index")
    if not gr.empty:
        gr = gr[gr.year_index <= n4x].dropna(subset=["dT_K", "dN_Wm2"])
        i = gr.year_index.to_numpy()
        blocks["abrupt-4xCO2/T"] = (i, gr.dT_K.to_numpy(), 1)
        blocks["abrupt-4xCO2/N"] = (i, gr.dN_Wm2.to_numpy(), 1)
    tr = _read("transient_1pct.csv")
    tr = tr[tr.source == source].sort_values("year_index").dropna(subset=["dT_K"])
    if not tr.empty:
        blocks["1pctCO2/T"] = (tr.year_index.to_numpy(), tr.dT_K.to_numpy(), 1)
    return blocks


def load(source, driver="concentration", tiers=None, n4x=N4X):
    noise = control_noise(source)
    for var, v in member_noise(source, driver).items():
        if not np.isfinite(noise[var][0]):  # no usable piControl series
            noise[var] = v
    blocks = {**_scenario_blocks(source, driver, tiers), **_idealised_blocks(source, n4x)}
    out = {}
    for key, (x, y, k) in blocks.items():
        var = key.split("/")[1]
        sigma, rho = noise[var]
        out[key] = {"x": x, "y": y, "k": k, "sigma_iv": sigma / np.sqrt(k), "rho": rho}
    return out


def _spread_noise(df, col, value_floor):
    """Single-member sigma, lag-1 rho from deviations about the ensemble mean (tiers with k>=3)."""
    dev, lag = [], []
    for _, x in df[df.year > HIST_END].drop_duplicates(["tier", "variant", "year"]).groupby("tier"):
        p = x.pivot_table(index="year", columns="variant", values=col).dropna(axis=1)
        k = p.shape[1]
        if k < 3:
            continue
        d = p.sub(p.mean(axis=1), axis=0).to_numpy() * np.sqrt(k / (k - 1))
        dev.append(d.ravel())
        lag.append(np.corrcoef(d[:-1].ravel(), d[1:].ravel())[0, 1])
    if not dev:
        return value_floor, 0.5
    return max(float(np.concatenate(dev).std()), value_floor), float(np.mean(lag))


def load_emissions(source):
    """Emissions-driven blocks: GSAT (T), CO2 concentration (C, ppm), ocean uptake (O, PgC/yr).

    History (1850-2021) is counted once; scenario tiers 2022-2100.
    """
    g = _read("gsat.csv")
    g = g[(g.ensemble == "ESM") & (g.source == source) & (g.driver == "emissions")]
    c = _read("atmospheric_co2.csv")
    c = c[(c.source == source) & (c.driver == "emissions")]
    o = _read("carbon_budget.csv")
    o = o[(o.ensemble == "ESM") & (o.source == source) & (o.driver == "emissions")]
    out = {}
    for var, df, col, floor in (("T", g, "gsat_raw_K", 0.05), ("C", c, "co2_ppm", 0.2), ("O", o, "ocean_PgC_yr", 0.05)):
        if df.empty:
            continue
        sigma, rho = _spread_noise(df, col, floor)
        hist = df[(df.year <= HIST_END)].drop_duplicates(["variant", "year"])
        fut = df[df.year > HIST_END].drop_duplicates(["tier", "variant", "year"])
        groups = [("hist", hist)] + [(t, x) for t, x in fut.groupby("tier") if t != "hist"]
        for tier, x in groups:
            s = x.groupby("year")[col].agg(["mean", "count"]).dropna()
            if s.empty:
                continue
            k = int(s["count"].median())
            out[f"{tier}/{var}"] = {"x": s.index.to_numpy(), "y": s["mean"].to_numpy(), "k": k,
                                    "sigma_iv": sigma / np.sqrt(k), "rho": rho}
    return out


def metrics(source):
    m = _read("metrics.csv")
    m = m[(m.ensemble == "ESM") & (m.source == source)]
    return dict(zip(m.metric_id, m.value))


if __name__ == "__main__":
    import sys

    for src in sys.argv[1:] or ["CanESM6-0-MR", "UKESM1-3-LL"]:
        print(src, control_noise(src), metrics(src))
        for k, b in load(src).items():
            print(f"  {k:18s} n={len(b['x']):3d} k={b['k']} x={b['x'][0]}..{b['x'][-1]} "
                  f"y_end={b['y'][-1]:.2f} sig={b['sigma_iv']:.3f} rho={b['rho']:.2f}")

#!/usr/bin/env python
"""Extract per-rung climate-sensitivity / carbon-cycle metrics for the
cumulative structural-ladder experiments and write a single tidy summary CSV
that the paper (Section 5) reads its numbers from.

The script is deliberately self-contained and depends only on numpy/pandas so
that it can live in the paper repo and be re-run against Marit's dumps (read
only) by absolute path, or copied verbatim into ``cscm-calibrate/scripts`` once
the remaining (+PDO, IPCC-on-PDO) dumps exist.

ECS (Gregory), TCR and TCRE are computed here directly from the idealised
scenario outputs, mirroring ``plot_and_evaluate_w_ar7_targets.py``
(compute_ecs_gregory / TCR-at-yr70 / TCRE-at-yr100) so that the figures and the
manuscript come from one code path. TCRE/ZEC/TPW from the flat10 pipeline are
read from the already-computed metric CSVs when present.

Usage:
    python extract_ladder_metrics.py            # prints table + writes CSV
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration: where Marit's dumps live, and the ladder definition.
# Edit DUMP_ROOT if the calibrate checkout moves.
# --------------------------------------------------------------------------
DUMP_ROOT = Path("/div/no-backup-nac/users/masan/GRAFITE/cscm-calibrate/scripts")
OUT_CSV = Path(__file__).resolve().parent / "ladder_metrics_summary.csv"

# Each rung: (label, dump-dir, flat10-metrics-suffix, status)
# flat10 suffix is the trailing token of flat10_ciceroscm_key_metrics_<suffix>.csv
LADDER = [
    ("0 baseline (v1)", "out_file_dump_nopattern_noefficacy", "nopattern_noefficacy", "done"),
    ("1 +efficacy",     "out_file_dump_nopattern",            "nopattern",            "done"),
    ("2 +pattern",      "out_file_dump",                      "pattern",              "done"),
    ("4a IPCC+pattern", "out_file_dump_ar7_v1",               None,                   "done"),
    # ("3 +PDO",            "out_file_dump_pdo",     "pdo",     "todo"),
    # ("4b IPCC+pat+PDO",   "out_file_dump_ar7_pdo", "pdo_ar7", "todo"),
]

TEMP_VAR = "Surface Air Ocean Blended Temperature Change"
OHC_VAR = "Heat Content|Ocean"
START_YEAR = 1850  # idealised perturbation start (first 100 yr are spinup)


# --------------------------------------------------------------------------
# Loading helpers (tolerant of both reformatted and raw draw_samples outputs).
# --------------------------------------------------------------------------
def load_scenario(dump: Path, scenario: str) -> pd.DataFrame | None:
    """Return the member x year output for *scenario*, trying the reformatted
    ``_rcmip_ciceroscm_`` file first and the raw ``draw_samples`` file second."""
    for pattern in (
        f"{scenario}_rcmip_ciceroscm_*.csv",
        f"{scenario}_*draw_samples*.csv",
        f"{scenario}_*.csv",
    ):
        hits = sorted(glob.glob(str(dump / pattern)))
        hits = [h for h in hits if "exceedance" not in h]
        if hits:
            return pd.read_csv(hits[0], index_col=0)
    return None


def timeseries(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    rows = df[df["variable"] == variable]
    year_cols = [c for c in df.columns if str(c).isdigit()]
    key = "ensemble_member" if "ensemble_member" in rows.columns else "run_id"
    ts = rows.set_index(key)[year_cols]
    ts.columns = ts.columns.astype(int)
    return ts.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------
# Metric computations.
# --------------------------------------------------------------------------
def ecs_gregory(df: pd.DataFrame, n_years: int = 150) -> pd.Series:
    """ECS for 2xCO2 via Gregory regression of annual OHC change on dT over the
    first *n_years* of abrupt-4xCO2."""
    temp = timeseries(df, TEMP_VAR)
    ohc = timeseries(df, OHC_VAR)
    years = sorted(temp.columns)
    i0 = years.index(START_YEAR) if START_YEAR in years else 0
    years = years[i0 : i0 + n_years]
    t = temp[years].values
    o = ohc[years].values
    dt = (t - t[:, [0]])[:, 1:]
    n = np.diff(o, axis=1)
    out = []
    for k in range(len(t)):
        x, y = dt[k], n[k]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            out.append(np.nan)
            continue
        slope, intercept = np.polyfit(x[m], y[m], 1)
        out.append(np.nan if slope >= 0 else -intercept / slope / 2.0)
    return pd.Series(out, name="ECS")


def transient(df: pd.DataFrame, offset: int) -> pd.Series:
    """Temperature anomaly relative to START_YEAR at START_YEAR+offset
    (offset=69 -> TCR from 1pctCO2; offset=99 -> TCRE-style from esm-flat10)."""
    ts = timeseries(df, TEMP_VAR)
    anom = ts.subtract(ts[START_YEAR], axis=0)
    yr = START_YEAR + offset
    if yr not in anom.columns:
        yr = min(anom.columns, key=lambda y: abs(y - yr))
    return anom[yr]


def eoc_warming(df: pd.DataFrame) -> pd.Series:
    """End-of-century warming: 2081-2100 mean minus the 1850-1900
    pre-industrial baseline."""
    ts = timeseries(df, TEMP_VAR)
    ref = [y for y in ts.columns if 1850 <= y <= 1900]
    fut = [y for y in ts.columns if 2081 <= y <= 2100]
    return ts[fut].mean(axis=1) - ts[ref].mean(axis=1)


# Forward scenarios for the end-of-century projection diagnostic
# (high and very-low, to bracket the scenario range).
PROJ_SCENARIOS = {"scen7H": "esm-allGHG-scen7-H", "scen7VL": "esm-allGHG-scen7-VL"}


def qstr(s: pd.Series) -> tuple[float, float, float, int]:
    s = s.dropna()
    return (s.median(), s.quantile(0.05), s.quantile(0.95), len(s))


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def flat10_metrics(suffix: str) -> pd.DataFrame | None:
    if suffix is None:
        return None
    f = DUMP_ROOT / "figures" / "flat10" / f"flat10_ciceroscm_key_metrics_{suffix}.csv"
    return pd.read_csv(f) if f.exists() else None


def main() -> None:
    records = []
    for label, dumpname, flat_suffix, status in LADDER:
        dump = DUMP_ROOT / dumpname
        row = {"rung": label, "dump": dumpname, "status": status}

        a4 = load_scenario(dump, "abrupt-4xCO2")
        if a4 is not None:
            row["ECS_med"], row["ECS_p05"], row["ECS_p95"], row["ECS_n"] = qstr(ecs_gregory(a4))
        p1 = load_scenario(dump, "1pctCO2")
        if p1 is not None:
            row["TCR_med"], row["TCR_p05"], row["TCR_p95"], row["TCR_n"] = qstr(transient(p1, 69))

        fm = flat10_metrics(flat_suffix)
        if fm is not None:
            for col, out in (("tcre", "TCRE"), ("zec50", "ZEC50"),
                             ("zec100", "ZEC100"), ("tpw", "TPW")):
                if col in fm:
                    m, lo, hi, n = qstr(fm[col])
                    row[f"{out}_med"], row[f"{out}_p05"], row[f"{out}_p95"] = m, lo, hi

        for tag, scen in PROJ_SCENARIOS.items():
            pdf = load_scenario(dump, scen)
            if pdf is not None:
                m, lo, hi, n = qstr(eoc_warming(pdf))
                row[f"W_{tag}_med"], row[f"W_{tag}_p05"], row[f"W_{tag}_p95"] = m, lo, hi
        records.append(row)

    summary = pd.DataFrame.from_records(records)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

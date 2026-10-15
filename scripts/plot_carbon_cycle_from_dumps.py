"""Carbon-cycle ensemble plots from the three CICERO-SCM dump folders.

Mirrors the structure of
  CSCM_temp/carbon_cycle_stuff/cscm_carbon_comparison_plots/plots_all_outline.py
but reads 500-member ensemble CSVs from the three dump folders rather than
running the model.  Each plot overlays all three sources as median lines with
5th–95th percentile shading.

Experiment → dump-scenario mapping
  Friedlingstein plots  → "historical"
  Terhaar plots         → "historical", "ssp585", "ssp245", "ssp126"
  Seferian plots        → "historical", "1pctCO2"

Variable derivation (variables not directly in the dump CSVs)
  Biosphere carbon flux   = annual diff of Carbon Pool|Land  (positive = land sink)
  Ocean carbon flux       = annual diff of Carbon Pool|Ocean (positive = ocean sink)
  Atmospheric carbon flux = Net Flux to Atmosphere|CO2  (pass-through)
  Emissions CO2           = bio + ocean + atmospheric    (budget identity)
  Airborne fraction CO2   = Atmospheric carbon flux / Emissions CO2
  Ocean heat uptake       = Heat Uptake (ZJ/yr, direct; fallback = diff Heat Content|Ocean)

Usage
-----
  python plot_carbon_cycle_from_dumps.py               # all plots, with obs data
  python plot_carbon_cycle_from_dumps.py --no-data     # model ensemble only
  python plot_carbon_cycle_from_dumps.py --skip-friedlingstein --skip-terhaar
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_various_from_experiments import (
    DUMP_NEW,
    DUMP_OLD,
    DUMP_WIDE
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

LOOK_AT_DATA_DIR = Path(
    "/div/no-backup-nac/users/masan/CSCM_temp/carbon_cycle_stuff"
    "/cscm_carbon_comparison_plots"
)

FIG_OUT = SCRIPT_DIR / "figures" / "carbon_cycle"

CSV_SUFFIX = {
    "pattern": "_rcmip_draw_samples_delta_aero_and_efficacy_wide_lambda_500.csv",
    "nopattern": "_rcmip_draw_samples_no_delta_aero_wide_lambda_400.csv",
    "nopattern_noefficacy": "_rcmip_draw_samples_no_efficacy_no_pattern_wide_lambda_400.csv",
}

# Source labels, dump-folder paths, colours and linestyles
SOURCE_STYLES: dict[str, tuple[Path, str, str]] = {
    "nopattern_noefficacy":  (DUMP_NEW,  "C0", "-"),
    "nopattern": (DUMP_OLD,  "C1", "-"),
    "pattern": (DUMP_WIDE, "C2", "-"),
}

# ---------------------------------------------------------------------------
# look_at_data import (uses relative data_temp/ paths, so we chdir before use)
# ---------------------------------------------------------------------------
_look_at_data_module = None


def _import_look_at_data():
    """Return the look_at_data module, importing it lazily on first call.

    Adds LOOK_AT_DATA_DIR to sys.path so the module can be found.  Returns
    None (with a warning) if the directory or module cannot be located.
    """
    global _look_at_data_module
    if _look_at_data_module is not None:
        return _look_at_data_module
    if not LOOK_AT_DATA_DIR.exists():
        warnings.warn(f"look_at_data directory not found: {LOOK_AT_DATA_DIR}")
        return None
    sys.path.insert(0, str(LOOK_AT_DATA_DIR))
    try:
        import look_at_data as _lad
        _look_at_data_module = _lad
        return _lad
    except ImportError as exc:
        warnings.warn(f"Could not import look_at_data: {exc}")
        return None


@contextlib.contextmanager
def _look_at_data_cwd():
    """Context manager: chdir to look_at_data directory so its relative
    data_temp/ paths resolve correctly, then restore cwd on exit."""
    old_cwd = os.getcwd()
    os.chdir(LOOK_AT_DATA_DIR)
    try:
        yield
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Low-level CSV helpers (adapted from plot_various_from_experiments.py)
# ---------------------------------------------------------------------------

def load_scenario_csv(folder: Path, scenario: str, runtype: str) -> pd.DataFrame | None:
    """Load a scenario draw-samples CSV, or return None if missing."""
    path = folder / f"{scenario}{CSV_SUFFIX[runtype]}"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def extract_ensemble(df: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (years_1d, array[n_ens, n_yr]) for *variable*, or None."""
    sub = df.loc[df["variable"] == variable]
    if sub.empty:
        return None
    arr = sub.iloc[:, 7:].to_numpy(dtype=float)
    year_cols = sub.columns[7:].astype(int).to_numpy()
    if variable == "Net Flux to Atmosphere|CO2":
        # In the dump, this variable is negative for net flux to atmosphere,
        # but we want it positive for plotting as "Atmospheric carbon flux".
        arr = -arr
    return year_cols, arr


def percentile_band(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p05, p50, p95) across ensemble axis."""
    return np.nanpercentile(arr, [5, 50, 95], axis=0)


def plot_band(
    ax,
    years: np.ndarray,
    p05: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    """Plot a shaded 5–95 % band with a solid median line on *ax*.

    The band uses *alpha* = 0.20; the median line uses *lw* = 1.5.
    """
    ax.fill_between(years, p05, p95, color=color, alpha=0.20, linewidth=0)
    ax.plot(years, p50, color=color, lw=1.5, linestyle=linestyle,
            label=f"{label} (median)")


# ---------------------------------------------------------------------------
# Variable derivation helper
# ---------------------------------------------------------------------------

def extract_or_derive(df: pd.DataFrame, var_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (years, array[n_ens, n_yr]) for *var_name*.

    If the variable is not directly present in *df*, it is derived from
    related pool or flux variables that are present.
    """
    # -- direct lookup first --
    direct = extract_ensemble(df, var_name)
    if direct is not None:
        return direct

    # -- derived variables --
    if var_name == "Biosphere carbon flux":
        land = extract_ensemble(df, "Carbon Pool|Land")
        if land is None:
            return None
        years, arr = land
        flux = np.concatenate(
            [np.zeros((arr.shape[0], 1)), np.diff(arr, axis=1)], axis=1
        )
        return years, flux

    if var_name == "Ocean carbon flux":
        ocean = extract_ensemble(df, "Carbon Pool|Ocean")
        if ocean is None:
            return None
        years, arr = ocean
        flux = np.concatenate(
            [np.zeros((arr.shape[0], 1)), np.diff(arr, axis=1)], axis=1
        )
        return years, flux

    if var_name == "Atmospheric carbon flux":
        return extract_ensemble(df, "Net Flux to Atmosphere|CO2")

    if var_name == "Emissions CO2":
        bio = extract_or_derive(df, "Biosphere carbon flux")
        ocn = extract_or_derive(df, "Ocean carbon flux")
        atm = extract_or_derive(df, "Atmospheric carbon flux")
        if bio is None or ocn is None or atm is None:
            return None
        years = bio[0]
        n = min(bio[1].shape[0], ocn[1].shape[0], atm[1].shape[0])
        return years, bio[1][:n] + ocn[1][:n] + atm[1][:n]

    if var_name == "Airborne fraction CO2":
        em = extract_or_derive(df, "Emissions CO2")
        atm = extract_or_derive(df, "Atmospheric carbon flux")
        if em is None or atm is None:
            return None
        years = em[0]
        n = min(em[1].shape[0], atm[1].shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            af = np.where(em[1][:n] != 0, atm[1][:n] / em[1][:n], np.nan)
        return years, af

    if var_name == "Concentrations CO2":
        return extract_ensemble(df, "Atmospheric Concentrations|CO2")

    if var_name in ("dT_glob", "Surface Air Temperature Change"):
        return extract_ensemble(df, "Surface Air Temperature Change")

    if var_name == "OHCTOT":
        return extract_ensemble(df, "Heat Content|Ocean")

    if var_name == "Ocean heat uptake":
        result = extract_ensemble(df, "Heat Uptake")
        if result is not None:
            return result
        # Fallback: diff of Heat Content|Ocean (ZJ → ZJ/yr)
        ohc = extract_ensemble(df, "Heat Content|Ocean")
        if ohc is None:
            return None
        years, arr = ohc
        flux = np.concatenate(
            [np.zeros((arr.shape[0], 1)), np.diff(arr, axis=1)], axis=1
        )
        return years, flux

    return None


# ---------------------------------------------------------------------------
# Multi-source loader
# ---------------------------------------------------------------------------

def load_all_sources(scenario: str) -> dict[str, pd.DataFrame]:
    """Load *scenario* from every dump folder that has the file.

    Returns ``{label: DataFrame}`` for present sources only.
    """
    result = {}
    for label, (folder, _color, _ls) in SOURCE_STYLES.items():
        df = load_scenario_csv(folder, scenario, label)
        if df is None:
            print(f"  [load] {label}: missing {scenario}")
        else:
            result[label] = df
    return result


def get_gcb_total_emissions(data_gcb: pd.DataFrame) -> np.ndarray | None:
    """Return total GCB emissions as fossil plus land-use emissions."""
    if "emissions_tot" in data_gcb.columns:
        return data_gcb["emissions_tot"].to_numpy(dtype=float)

    fossil_keys = [
        "fossil emissions",
        "fossil emissions excluding carbonation",
    ]
    luc_key = "land-use change emissions"

    fossil = None
    for key in fossil_keys:
        if key in data_gcb.columns:
            fossil = data_gcb[key].to_numpy(dtype=float)
            break

    if fossil is None or luc_key not in data_gcb.columns:
        return None

    total = fossil + data_gcb[luc_key].to_numpy(dtype=float)
    if (
        "cement carbonation sink" in data_gcb.columns
        and "fossil emissions" not in data_gcb.columns
    ):
        total = total + data_gcb["cement carbonation sink"].to_numpy(dtype=float)
    return total


# ---------------------------------------------------------------------------
# Friedlingstein plots  (uses "historical" scenario)
# ---------------------------------------------------------------------------

def _make_friedlingstein_one(sources_data: dict, options: dict) -> None:
    """Figure 1 — atmospheric CO2 concentration 1960–2025."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, (_, color, ls) in SOURCE_STYLES.items():
        df = sources_data.get(label)
        if df is None:
            continue
        result = extract_or_derive(df, "Concentrations CO2")
        if result is None:
            print(f"  [Fig1] {label}: Concentrations CO2 not available, skipping")
            continue
        years, arr = result
        p05, p50, p95 = percentile_band(arr)
        plot_band(ax, years, p05, p50, p95, color=color, label=label, linestyle=ls)

    if options.get("include_data"):
        lad = _import_look_at_data()
        if lad is not None:
            try:
                with _look_at_data_cwd():
                    yearly = lad.read_noaa_gml_ml_means("year")
                    monthly = lad.read_noaa_gml_ml_means("month")
                ax.plot(yearly[0], yearly[1], color="red", lw=1.5, label="NOAA/GML (annual)")
                ax.plot(monthly[0], monthly[1], color="red", alpha=0.4, lw=0.8,
                        label="NOAA/GML (monthly)")
            except Exception as exc:
                warnings.warn(f"Could not load NOAA GML data: {exc}")

    ax.set_xlim(1960, 2025)
    ax.set_ylim(300, 420)
    ax.set_xlabel("Year")
    ax.set_ylabel("Atmospheric CO2 concentration (ppm)")
    ax.set_title("a)", fontsize=11, loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{options['filepath_start']}_Friedlingstein1.png", dpi=120)
    plt.close(fig)
    print(f"  saved {options['filepath_start']}_Friedlingstein1.png")


def _make_friedlingstein_three(sources_data: dict, options: dict,
                                data_gcb=None) -> None:
    """Figure 3 — carbon budget flux and cumulative.

    One column per CSCM dump, plus a GCB column when observed data are available.
    """
    components = [
        ("Biosphere carbon flux",   "green",  "Land sink"),
        ("Ocean carbon flux",       "blue",   "Ocean sink"),
        ("Atmospheric carbon flux", "orange", "Atmospheric growth"),
    ]

    include_gcb = options.get("include_data") and data_gcb is not None
    source_entries = [
        (label, sources_data[label], ls)
        for label, (_, _src_color, ls) in SOURCE_STYLES.items()
        if label in sources_data
    ]
    n_model_cols = len(source_entries)
    ncols = n_model_cols + (1 if include_gcb else 0)
    fig, axs = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(6 * ncols, 8),
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")

    # ----- Model panels -----
    for col_idx, (label, df, ls) in enumerate(source_entries):
        ax_flux_m = axs[0, col_idx]
        ax_cum_m = axs[1, col_idx]

        for comp_var, comp_color, comp_label in components:
            result = extract_or_derive(df, comp_var)
            if result is None:
                continue
            years, arr = result
            p05, p50, p95 = percentile_band(arr)
            # Use component colour with source alpha variation
            ax_flux_m.fill_between(years, p05, p95, color=comp_color, alpha=0.12, linewidth=0)
            ax_flux_m.plot(
                years,
                p50,
                color=comp_color,
                lw=1.0,
                linestyle=ls,
                alpha=0.85,
                label=comp_label,
            )
            cum_p05 = np.cumsum(p05)
            cum_p50 = np.cumsum(p50)
            cum_p95 = np.cumsum(p95)
            ax_cum_m.fill_between(years, cum_p05, cum_p95,
                                  color=comp_color, alpha=0.12, linewidth=0)
            ax_cum_m.plot(years, cum_p50, color=comp_color, lw=1.0,
                          linestyle=ls, alpha=0.85)

        # Overlay negative emissions
        em_result = extract_or_derive(df, "Emissions CO2")
        if em_result is not None:
            years_em, em_arr = em_result
            em_p05, em_p50, em_p95 = percentile_band(em_arr)
            ax_flux_m.fill_between(
                years_em,
                0,
                -em_p50,
                color="brown",
                alpha=0.20,
                linewidth=0,
                label="Total emissions (neg.)",
            )
            ax_flux_m.plot(
                years_em,
                -em_p50,
                color="brown",
                lw=1.0,
                linestyle=ls,
                alpha=0.85,
            )
            ax_cum_m.fill_between(
                years_em,
                0,
                -np.cumsum(em_p50),
                color="brown",
                alpha=0.20,
                linewidth=0,
            )
            ax_cum_m.plot(
                years_em,
                -np.cumsum(em_p50),
                color="brown",
                lw=1.0,
                linestyle=ls,
                alpha=0.85,
            )

        flux_letter = next(panel_letters)
        cum_letter = next(panel_letters)
        ax_flux_m.set_title(f"{flux_letter}) {label} — fluxes", fontsize=11, loc="left")
        ax_cum_m.set_title(f"{cum_letter}) {label} — cumulative", fontsize=11, loc="left")
        ax_flux_m.set_xlim(1850, 2020)
        ax_flux_m.grid(True, alpha=0.3)
        ax_cum_m.grid(True, alpha=0.3)
        ax_flux_m.set_xlabel("Year")
        ax_cum_m.set_xlabel("Year")

    axs[0, 0].set_ylabel("Carbon flux (Pg C / yr)")
    axs[1, 0].set_ylabel("Cumulative carbon (Pg C)")
    axs[0, 0].legend(fontsize=7)

    # ----- GCB panels -----
    if include_gcb:
        ax_flux_gcb = axs[0, n_model_cols]
        ax_cum_gcb = axs[1, n_model_cols]
        gcb_components = [
            ("land sink",           "green",  "Land sink"),
            ("ocean sink",          "blue",   "Ocean sink"),
            ("atmospheric growth",  "orange", "Atm. growth"),
        ]
        years_gcb = data_gcb["Year"].to_numpy()
        for gcb_key, gcb_color, gcb_label in gcb_components:
            if gcb_key in data_gcb.columns:
                vals = data_gcb[gcb_key].to_numpy(dtype=float)
                ax_flux_gcb.fill_between(years_gcb, 0, vals, color=gcb_color,
                                         alpha=0.4, label=gcb_label)
                ax_cum_gcb.fill_between(years_gcb, 0, np.cumsum(vals),
                                        color=gcb_color, alpha=0.4, label=gcb_label)
        vals_em = get_gcb_total_emissions(data_gcb)
        if vals_em is not None:
            ax_flux_gcb.fill_between(years_gcb, 0, -vals_em, color="brown",
                                     alpha=0.4, label="Total emissions (neg.)")
            ax_cum_gcb.fill_between(years_gcb, 0, -np.cumsum(vals_em),
                                    color="brown", alpha=0.4)
        ax_flux_gcb.set_ylabel("Carbon flux (Pg C / yr)")
        ax_cum_gcb.set_ylabel("Cumulative carbon (Pg C)")
        flux_letter = next(panel_letters)
        cum_letter = next(panel_letters)
        ax_flux_gcb.set_title(f"{flux_letter}) GCB observed — fluxes", fontsize=11, loc="left")
        ax_cum_gcb.set_title(f"{cum_letter}) GCB observed — cumulative", fontsize=11, loc="left")
        ax_flux_gcb.legend(fontsize=7)
        ax_flux_gcb.set_xlabel("Year")
        ax_cum_gcb.set_xlabel("Year")
        ax_flux_gcb.set_xlim(1850, 2020)
        ax_flux_gcb.set_ylim(-12, 12)
        ax_cum_gcb.set_ylim(-800, 800)

    fig.tight_layout()
    fig.savefig(f"{options['filepath_start']}_Friedlingstein3.png", dpi=120)
    plt.close(fig)
    print(f"  saved {options['filepath_start']}_Friedlingstein3.png")


def _make_friedlingstein_four(sources_data: dict, options: dict,
                               data_gcb=None) -> None:
    """Figure 4 — six individual flux-component panels."""
    variable_order = [
        ("Emissions CO2",           "Fossil + LUC Emissions",    "emissions_tot"),
        ("Atmospheric carbon flux", "Atmospheric Growth Rate",   "atmospheric growth"),
        ("Land Use Emissions",      "Land Use Emissions",        "land-use change emissions"),
        ("Biosphere carbon flux",   "Land Sink",                 "land sink"),
        ("budget imbalance",        "Budget Imbalance",          "budget imbalance"),
        ("Ocean carbon flux",       "Ocean Sink",                "ocean sink"),
    ]

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(10, 10))

    for i, (var_name, title, gcb_key) in enumerate(variable_order):
        ax = axs[i // 2, i % 2]
        ax.set_title(f"{chr(i + 97)})", fontsize=11, loc="left")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("CO2 Flux (Pg C / yr)")
        ax.grid(True, alpha=0.3)

        for label, (_, color, ls) in SOURCE_STYLES.items():
            df = sources_data.get(label)
            if df is None:
                continue

            if var_name == "Land Use Emissions":
                # Not in the dump; show zero line
                result = extract_or_derive(df, "Emissions CO2")
                if result is not None:
                    years, arr = result
                    ax.plot(years, np.zeros(len(years)), color=color,
                            lw=1.0, linestyle=ls, label=f"{label} (zero)")
                continue

            if var_name == "budget imbalance":
                em = extract_or_derive(df, "Emissions CO2")
                bio = extract_or_derive(df, "Biosphere carbon flux")
                ocn = extract_or_derive(df, "Ocean carbon flux")
                atm = extract_or_derive(df, "Atmospheric carbon flux")
                if any(x is None for x in [em, bio, ocn, atm]):
                    continue
                years = em[0]
                n = min(em[1].shape[0], bio[1].shape[0],
                        ocn[1].shape[0], atm[1].shape[0])
                imbal = em[1][:n] - atm[1][:n] - bio[1][:n] - ocn[1][:n]
                p05, p50, p95 = percentile_band(imbal)
                plot_band(ax, years, p05, p50, p95, color=color, label=label, linestyle=ls)
                continue

            result = extract_or_derive(df, var_name)
            if result is None:
                print(f"  [Fig4] {label}: {var_name} not available, skipping")
                continue
            years, arr = result
            p05, p50, p95 = percentile_band(arr)
            plot_band(ax, years, p05, p50, p95, color=color, label=label, linestyle=ls)

        if options.get("include_data") and data_gcb is not None:
            if gcb_key in data_gcb.columns:
                gcb_years = data_gcb["Year"].to_numpy()
                gcb_vals  = data_gcb[gcb_key].to_numpy(dtype=float)
                ax.plot(gcb_years, gcb_vals, color="red", alpha=0.6,
                        lw=1.2, label="GCB observed")

    axs[0, 0].legend(fontsize=7)
    axs[2, 1].set_xlim(1850, 2020)
    fig.tight_layout()
    fig.savefig(f"{options['filepath_start']}_Friedlingstein4.png", dpi=120)
    plt.close(fig)
    print(f"  saved {options['filepath_start']}_Friedlingstein4.png")


def make_friedlingstein_plots(options: dict) -> None:
    """Load 'historical' from all dump sources and produce Friedlingstein figures 1, 3, 4."""
    print("[Friedlingstein] loading historical …")
    sources_data = load_all_sources("esm-hist")
    if not sources_data:
        print("[Friedlingstein] no data found, skipping")
        return

    data_gcb = None
    if options.get("include_data"):
        lad = _import_look_at_data()
        if lad is not None:
            try:
                with _look_at_data_cwd():
                    data_gcb = lad.read_gcb_data()
            except Exception as exc:
                warnings.warn(f"Could not load GCB data: {exc}")

    _make_friedlingstein_one(sources_data, options)
    _make_friedlingstein_three(sources_data, options, data_gcb=data_gcb)
    _make_friedlingstein_four(sources_data, options, data_gcb=data_gcb)


# ---------------------------------------------------------------------------
# Terhaar plots  (uses "historical" + "ssp585", "ssp245", "ssp126")
# ---------------------------------------------------------------------------

SSP_COLOURS = {
    "ssp585": "red",
    "ssp245": "orange",
    "ssp126": "blue",
}
SSP_LABELS = {
    "ssp585": "SSP5-8.5",
    "ssp245": "SSP2-4.5",
    "ssp126": "SSP1-2.6",
}
SOURCE_LINESTYLES = {
    "nopattern":  "-",
    "nopattern_noefficacy": "--",
    "pattern":       ":",
}


def make_terhaar_plots(options: dict) -> None:
    """Produce Terhaar figure 1 (two panels).

    Panel a: ocean carbon flux from 'historical', 1990–2020.
    Panel b: cumulative ocean carbon flux for ssp585, ssp245, ssp126, 1950–2100.
    """
    print("[Terhaar] loading data …")
    sources_hist = load_all_sources("esm-hist")

    ssp_data: dict[str, dict[str, pd.DataFrame]] = {}
    for ssp in SSP_COLOURS:
        ssp_data[ssp] = load_all_sources(ssp)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax_hist, ax_ssp = axs

    # ----- Panel a: historical ocean carbon flux -----
    for label, (_, _col, _ls) in SOURCE_STYLES.items():
        df = sources_hist.get(label)
        if df is None:
            continue
        result = extract_or_derive(df, "Ocean carbon flux")
        if result is None:
            continue
        years, arr = result
        p05, p50, p95 = percentile_band(arr)
        plot_band(ax_hist, years, p05, p50, p95,
                  color=_col, label=label, linestyle=SOURCE_LINESTYLES[label])

    if options.get("include_data"):
        lad = _import_look_at_data()
        if lad is not None:
            try:
                with _look_at_data_cwd():
                    gcb_ocean = lad.read_gcb_ocean_carbon_data()
                if "year" in gcb_ocean.columns and "GCB" in gcb_ocean.columns:
                    ax_hist.plot(gcb_ocean["year"], gcb_ocean["GCB"],
                                 color="black", lw=1.5, label="GCB")
                if "Multi-model mean" in gcb_ocean.columns:
                    mm = gcb_ocean["Multi-model mean"].to_numpy(dtype=float)
                    yr = gcb_ocean["year"].to_numpy()
                    ax_hist.plot(yr, mm, color="green", lw=1.5, label="GCB multi-model")
                    if "Model Spread (sd)" in gcb_ocean.columns:
                        sd = gcb_ocean["Model Spread (sd)"].to_numpy(dtype=float)
                        ax_hist.fill_between(yr, mm - sd, mm + sd,
                                             color="green", alpha=0.2)
            except Exception as exc:
                warnings.warn(f"Could not load GCB ocean data: {exc}")

    ax_hist.set_xlim(1990, 2020)
    ax_hist.set_xlabel("Year")
    ax_hist.set_ylabel("Ocean C$_{ant}$ uptake (Pg C yr$^{-1}$)")
    ax_hist.set_title("a)", fontsize=11, loc="left")
    ax_hist.legend(fontsize=7)
    ax_hist.grid(True, alpha=0.3)

    # ----- Panel b: cumulative ocean carbon flux for SSPs -----
    for ssp, ssp_color in SSP_COLOURS.items():
        for label, (_, _col, _ls) in SOURCE_STYLES.items():
            df = ssp_data[ssp].get(label)
            if df is None:
                continue
            result = extract_or_derive(df, "Ocean carbon flux")
            if result is None:
                continue
            years, arr = result
            p05, p50, p95 = percentile_band(arr)
            cum_p05 = np.cumsum(p05)
            cum_p50 = np.cumsum(p50)
            cum_p95 = np.cumsum(p95)
            ax_ssp.fill_between(years, cum_p05, cum_p95,
                                 color=ssp_color, alpha=0.08, linewidth=0)
            ax_ssp.plot(years, cum_p50, color=ssp_color, lw=1.2,
                        linestyle=SOURCE_LINESTYLES[label],
                        label=f"{SSP_LABELS[ssp]} [{label}]")

    if options.get("include_data"):
        ax_ssp.errorbar(2010, 155, yerr=31, fmt="o", capsize=4,
                        color="black", label="Khatiwala et al. 2013")

    ax_ssp.set_xlim(1950, 2100)
    ax_ssp.set_xlabel("Year")
    ax_ssp.set_ylabel("Cumulative ocean C$_{ant}$ uptake (Pg C)")
    ax_ssp.set_title("b)", fontsize=11, loc="left")
    ax_ssp.legend(fontsize=6, ncol=2)
    ax_ssp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{options['filepath_start']}_Terhaar1.png", dpi=120)
    plt.close(fig)
    print(f"  saved {options['filepath_start']}_Terhaar1.png")


# ---------------------------------------------------------------------------
# Seferian plots  (uses "historical" + "1pctCO2")
# ---------------------------------------------------------------------------

# Column offset within the dump CSVs at which year 1850 sits.
# Year columns start at 1750 (column index 7 in the DataFrame), so
# 1850 = offset 100, 2000 = offset 250.
_1PCT_START_IDX = 100   # year 1850 = start of 1pctCO2 experiment
_1PCT_END_IDX   = 250   # year 1999 = end of 150-year segment to plot

SEFERIAN_VARS = {
    "dT_glob":          ("Surface Air Temperature Change",
                         "GMST (K)",        "tas"),
    "Ocean heat uptake": ("Ocean heat uptake",
                          "Heat uptake (ZJ yr$^{-1}$)", "hfds"),
    "Ocean carbon flux": ("Ocean carbon flux",
                          "Ocean C uptake (Pg C yr$^{-1}$)", "fgco2"),
}


def _make_seferian_two(sources_hist: dict, options: dict) -> None:
    """Seferian figure 2 — historical time-series for three variables."""
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))

    for i, (short_name, (var_name, ylabel, cmip6_var)) in enumerate(
        SEFERIAN_VARS.items()
    ):
        ax = axs[i]
        for label, (_, color, ls) in SOURCE_STYLES.items():
            df = sources_hist.get(label)
            if df is None:
                continue
            result = extract_or_derive(df, var_name)
            if result is None:
                print(f"  [Sef2] {label}: {var_name} not available, skipping")
                continue
            years, arr = result
            p05, p50, p95 = percentile_band(arr)
            plot_band(ax, years, p05, p50, p95, color=color, label=label, linestyle=ls)

        if options.get("include_data"):
            lad = _import_look_at_data()
            if lad is not None:
                try:
                    with _look_at_data_cwd():
                        lad.add_cmip6_lines_to_plot("historical", cmip6_var, ax)
                except Exception as exc:
                    warnings.warn(f"[Sef2] CMIP6 lines ({cmip6_var}): {exc}")
                if var_name == "Ocean carbon flux":
                    try:
                        with _look_at_data_cwd():
                            gcb_ocean = lad.read_gcb_ocean_carbon_data()
                        if "year" in gcb_ocean.columns and "GCB" in gcb_ocean.columns:
                            ax.plot(gcb_ocean["year"], gcb_ocean["GCB"],
                                    color="black", lw=1.5, label="GCB")
                        if "Multi-model mean" in gcb_ocean.columns:
                            mm = gcb_ocean["Multi-model mean"].to_numpy(dtype=float)
                            yr = gcb_ocean["year"].to_numpy()
                            ax.plot(yr, mm, color="green", lw=1.5, label="GCB multi-model")
                            if "Model Spread (sd)" in gcb_ocean.columns:
                                sd = gcb_ocean["Model Spread (sd)"].to_numpy(dtype=float)
                                ax.fill_between(yr, mm - sd, mm + sd,
                                                color="green", alpha=0.2)
                    except Exception as exc:
                        warnings.warn(f"[Sef2] GCB ocean data: {exc}")

        ax.set_ylabel(ylabel, size=13)
        ax.set_title(f"{chr(i + 97)})", fontsize=11, loc="left")
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlim(1850, 2015)
    axs[-1].set_xlabel("Year", size=13)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{options['filepath_start']}_Seferian2.png", dpi=120)
    plt.close(fig)
    print(f"  saved {options['filepath_start']}_Seferian2.png")


def _make_seferian_three(sources_1pct: dict, options: dict) -> None:
    """Seferian figure 3 — 1pctCO2 experiment panels.

    Top-left, top-right, bottom-left: same three variables as Seferian 2
    but for the 1pctCO2 experiment over model years 0–150 (calendar 1850–1999).
    Bottom-right: scatter of ocean heat uptake vs ocean carbon flux.
    """
    var_list = list(SEFERIAN_VARS.items())  # 3 items
    model_years = np.arange(_1PCT_END_IDX - _1PCT_START_IDX)  # 0 … 149

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    for i, (short_name, (var_name, ylabel, cmip6_var)) in enumerate(var_list):
        ax = axs[i // 2, i % 2]

        for label, (_, color, ls) in SOURCE_STYLES.items():
            df = sources_1pct.get(label)
            if df is None:
                continue
            result = extract_or_derive(df, var_name)
            if result is None:
                print(f"  [Sef3] {label}: {var_name} not available, skipping")
                continue
            _years, arr = result
            # Slice to 1pctCO2 experiment window (columns 100:250 of year arrays)
            arr_slice = arr[:, _1PCT_START_IDX:_1PCT_END_IDX]
            p05, p50, p95 = percentile_band(arr_slice)
            startyr = _years[_1PCT_START_IDX]
            plot_band(ax, _years[_1PCT_START_IDX:_1PCT_END_IDX]-startyr, p05, p50, p95,
                      color=color, label=label, linestyle=ls)

        if options.get("include_data"):
            lad = _import_look_at_data()
            if lad is not None:
                try:
                    with _look_at_data_cwd():
                        lad.add_cmip6_lines_to_plot("1pctCO2", cmip6_var, ax)
                    with _look_at_data_cwd():
                        lad.add_cmip6_lines_to_plot("1pctCO2-bgc", cmip6_var, ax, cl="pink")
                except Exception as exc:
                    warnings.warn(f"[Sef3] CMIP6 lines ({cmip6_var}): {exc}")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model time (yr; 0 = 1850)")
        ax.set_xlim(0, 150)
        ax.set_title(f"{chr(i + 97)})", fontsize=11, loc="left")
        ax.grid(True, alpha=0.3)

    # ----- Bottom-right: scatter heat uptake vs carbon flux -----
    ax_scatter = axs[1, 1]
    for label, (_, color, ls) in SOURCE_STYLES.items():
        df = sources_1pct.get(label)
        if df is None:
            continue
        h_res = extract_or_derive(df, "Ocean heat uptake")
        c_res = extract_or_derive(df, "Ocean carbon flux")
        if h_res is None or c_res is None:
            continue
        _yh, h_arr = h_res
        _yc, c_arr = c_res
        n = min(h_arr.shape[0], c_arr.shape[0])
        h_slice = h_arr[:n, _1PCT_START_IDX:_1PCT_END_IDX]
        c_slice = c_arr[:n, _1PCT_START_IDX:_1PCT_END_IDX]
        # Faint individual members
        for j in range(n):
            ax_scatter.plot(h_slice[j], c_slice[j], alpha=0.05, lw=0.5, color=color)
        # Bold median
        h_med = np.nanmedian(h_slice, axis=0)
        c_med = np.nanmedian(c_slice, axis=0)
        ax_scatter.plot(h_med, c_med, color=color, lw=2.0, label=f"{label} (median)")

    if options.get("include_data"):
        lad = _import_look_at_data()
        if lad is not None:
            try:
                with _look_at_data_cwd():
                    lad.add_cmip6_lines_combined_hfds_fgco2(ax_scatter)
                with _look_at_data_cwd():
                    lad.add_cmip6_lines_combined_hfds_fgco2(
                        ax_scatter, exp="1pctCO2-bgc", cl="pink"
                    )
            except Exception as exc:
                warnings.warn(f"[Sef3] CMIP6 scatter lines: {exc}")

    ax_scatter.set_xlabel("Heat uptake (ZJ yr$^{-1}$)")
    ax_scatter.set_ylabel("Ocean C uptake (Pg C yr$^{-1}$)")
    ax_scatter.set_title("d)", fontsize=11, loc="left")
    ax_scatter.legend(fontsize=7)
    ax_scatter.grid(True, alpha=0.3)

    seferian_handles, seferian_labels = axs[0, 1].get_legend_handles_labels()
    seferian_handles.extend([
        Line2D([0], [0], color="grey", lw=2, label="CMIP6 1pctCO2"),
        Line2D([0], [0], color="pink", lw=2, label="CMIP6 1pctCO2-bgc"),
    ])
    seferian_labels.extend(["CMIP6 1pctCO2", "CMIP6 1pctCO2-bgc"])
    axs[0, 1].legend(seferian_handles, seferian_labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{options['filepath_start']}_Seferian3.png", dpi=120)
    plt.close(fig)
    print(f"  saved {options['filepath_start']}_Seferian3.png")


def make_seferian_plots(options: dict) -> None:
    """Load 'historical' and '1pctCO2' and produce Seferian figures 2 and 3."""
    print("[Seferian] loading historical …")
    sources_hist = load_all_sources("historical")
    print("[Seferian] loading 1pctCO2 …")
    sources_1pct = load_all_sources("1pctCO2")

    if sources_hist:
        _make_seferian_two(sources_hist, options)
    else:
        print("[Seferian] no historical data, skipping Seferian 2")

    if sources_1pct:
        _make_seferian_three(sources_1pct, options)
    else:
        print("[Seferian] no 1pctCO2 data, skipping Seferian 3")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and dispatch to the requested plot groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-friedlingstein", action="store_true",
                        help="Skip Friedlingstein figures 1, 3, 4")
    parser.add_argument("--skip-terhaar", action="store_true",
                        help="Skip Terhaar figure 1")
    parser.add_argument("--skip-seferian", action="store_true",
                        help="Skip Seferian figures 2 and 3")
    parser.add_argument("--no-data", action="store_true",
                        help="Disable observational/CMIP6 overlays "
                             "(model ensemble only)")
    args = parser.parse_args()

    FIG_OUT.mkdir(parents=True, exist_ok=True)

    filepath_base = str(FIG_OUT / "cscm_ensemble")
    include_data  = not args.no_data

    options = {
        "filepath_start": filepath_base,
        "include_data":   include_data,
    }

    if not args.skip_friedlingstein:
        make_friedlingstein_plots(options)

    if not args.skip_terhaar:
        make_terhaar_plots(options)

    if not args.skip_seferian:
        make_seferian_plots(options)

    print(f"\nDone. Figures written to {FIG_OUT}")


if __name__ == "__main__":
    main()

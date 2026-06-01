"""Compare CICERO-SCM ensemble outputs between two dump folders and replicate
flat10_trials analyses.

Block A: For every experiment present in both ``out_file_dump/`` and
``out_file_dump_new_preCO2/``, plot every variable as a single panel showing
median + 5-95 percentile plume for both sources overlaid (blue = old,
orange = new preCO2).

Block B: Replicate the analyses from /div/nac/users/masan/flat10_trials/
(time-series multi-scenario plots, key metrics CSV, pairplot/histograms,
Gregory plots) using ``out_file_dump_new_preCO2/`` data on the flat10 class
of experiments.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DUMP_WIDE = SCRIPT_DIR / "out_file_dump"
DUMP_OLD = SCRIPT_DIR / "out_file_dump_nopattern"
DUMP_NEW = SCRIPT_DIR / "out_file_dump_nopattern_noefficacy"
FIG_ROOT = SCRIPT_DIR / "figures"
FIG_COMPARE = FIG_ROOT / "comparisons"
FIG_COMPARE_BY_VAR = FIG_COMPARE / "by_variable"
FIG_FLAT10 = FIG_ROOT / "flat10"

CSV_SUFFIX = {
    "pattern": "_rcmip_draw_samples_delta_aero_and_efficacy_wide_lambda_500.csv",
    "nopattern": "_rcmip_draw_samples_no_delta_aero_wide_lambda_400.csv",
    "nopattern_noefficacy": "_rcmip_draw_samples_no_efficacy_no_pattern_wide_lambda_400.csv",
}

YEAR_START = 1750
YEAR_END = 2169  # inclusive
YEARS_FULL = np.arange(YEAR_START, YEAR_END + 1)

# Variables present in the dump CSVs
VARIABLES = [
    "Carbon Pool|Land",
    "Carbon Pool|Ocean",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Anthropogenic|CO2",
    "Net Flux to Atmosphere|CO2",
    "Heat Content|Ocean",
    "Heat Content|Ocean|0-700m",
    "Heat Uptake",
    "Surface Air Ocean Blended Temperature Change",
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
]

# Replicates the variable dict from flat10_trials/make_timeseries_plots.py.
# Variables not in the CICERO-SCM dump are skipped at runtime with a warning.
FLAT10_VARIABLES = {
    "temp": ("Surface Air Temperature Change", "Temperature change (K)"),
    "co2_conc": ("Atmospheric Concentrations|CO2", "CO2 concentrations ppmv"),
    "co2_em": ("Emissions|CO2", "CO2 emissions (Pg_C yr-1)"),
    "erf": ("Effective Radiative Forcing", "Effective Radiative Forcing (W m-2)"),
    "OHC": ("Heat Content|Ocean", "Ocean heat content (ZJ)"),
    "biocarbon": ("Biosphere carbon flux", "Biosphere carbon flux (Pg_C yr-1)"),
    "oceancarbon": ("Ocean carbon flux", "Ocean carbon flux (Pg_C yr-1)"),
    "erf_co2": ("Effective Radiative Forcing|CO2", "ERF CO2 (W m-2)"),
    "erf_ch4": ("Effective Radiative Forcing|CH4", "ERF CH4 (W m-2)"),
    "erf_n2o": ("Effective Radiative Forcing|N2O", "ERF N2O (W m-2)"),
    "erf_aer": ("Effective Radiative Forcing|Aerosols", "ERF Aerosols (W m-2)"),
    "erf_fgas": ("Effective Radiative Forcing|F-Gases", "ERF F-Gases (W m-2)"),
    "erf_strath2o": (
        "Effective Radiative Forcing|Stratospheric Water Vapor",
        "ERF Strat. H2O (W m-2)",
    ),
    "erf_strato3": (
        "Effective Radiative Forcing|Stratospheric Ozone",
        "ERF Strat. O3 (W m-2)",
    ),
    "erf_tropo3": (
        "Effective Radiative Forcing|Tropospheric Ozone",
        "ERF Trop. O3 (W m-2)",
    ),
    # Extras useful from CICERO-SCM output
    "carbon_land": ("Carbon Pool|Land", "Land carbon pool (Pg C)"),
    "carbon_ocean": ("Carbon Pool|Ocean", "Ocean carbon pool (Pg C)"),
    "net_flux_co2": ("Net Flux to Atmosphere|CO2", "Net Flux to Atmosphere CO2 (Pg C / yr)"),
    "heat_uptake": ("Heat Uptake", "Heat Uptake (ZJ / yr)"),
}

# Flat10 scenarios (file stems) for the replication block.
FLAT10_SCENARIOS = [
    "esm-flat10",
    "esm-flat10-zec",
    "esm-flat10-cdr",
    "esm-piControl",
]
ABRUPT_SCENARIOS = ["abrupt-2xCO2", "abrupt-4xCO2", "abrupt-0p5xCO2"]

# Colours used in flat10_trials/make_timeseries_plots.py
FLAT10_COLOURS = ["grey", "blue", "red", "green", "orange", "brown", "magenta"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Convert a variable name into a filesystem-safe slug."""
    s = re.sub(r"[|/]+", "__", name)
    s = re.sub(r"[^\w.+-]+", "_", s)
    return s.strip("_")


def list_scenarios(folder: Path, label:str) -> set[str]:
    """Return the set of scenario stems available in ``folder``."""
    out = set()
    for p in folder.glob(f"*{CSV_SUFFIX[label]}"):
        if "_exceedance_years" in p.name:
            continue
        stem = p.name[: -len(CSV_SUFFIX[label])]
        out.add(stem)
    return out


def load_scenario_csv(folder: Path, scenario: str, label: str) -> pd.DataFrame | None:
    """Load a scenario CSV (or return None if missing)."""
    path = folder / f"{scenario}{CSV_SUFFIX[label]}"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def extract_ensemble(df: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (years, array[n_ens, n_yr]) for ``variable`` from a long-form
    dump CSV, or None if the variable is absent."""
    sub = df.loc[df["variable"] == variable]
    if sub.empty:
        return None
    arr = sub.iloc[:, 7:].to_numpy(dtype=float)
    year_cols = sub.columns[7:].astype(int).to_numpy()
    return year_cols, arr


def percentile_band(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (p05, p50, p95) along axis 0."""
    p05, p50, p95 = np.nanpercentile(arr, [5, 50, 95], axis=0)
    return p05, p50, p95


def plot_band(ax, years, p05, p50, p95, *, color: str, label: str) -> None:
    ax.fill_between(years, p05, p95, color=color, alpha=0.25, linewidth=0)
    ax.plot(years, p50, color=color, lw=1.5, label=f"{label} (median)")


# ---------------------------------------------------------------------------
# Phase 2: comparison plots
# ---------------------------------------------------------------------------
def make_comparison_plots(resume: bool = False) -> None:
    FIG_COMPARE.mkdir(parents=True, exist_ok=True)
    FIG_COMPARE_BY_VAR.mkdir(parents=True, exist_ok=True)

    old = list_scenarios(DUMP_OLD, "nopattern")
    new = list_scenarios(DUMP_NEW, "nopattern_noefficacy")
    wide = list_scenarios(DUMP_WIDE, "pattern")
    common = sorted(old & new & wide)
    only_old = sorted(old - new - wide)
    only_new = sorted(new - old - wide)
    only_wide = sorted(wide - old - new)

    print(f"[comparisons] {len(common)} scenarios in both dumps")
    if only_old:
        print(f"[comparisons]   only in old: {only_old}")
    if only_new:
        print(f"[comparisons]   only in new: {only_new}")
    if only_wide:
        print(f"[comparisons]   only in wide: {only_wide}")
    if resume:
        print("[comparisons] resume mode: skipping figures that already exist")

    for i, scenario in enumerate(common, 1):
        print(f"[comparisons] ({i}/{len(common)}) {scenario}")
        df_old = None
        df_new = None
        df_wide = None

        # Lazy-load so a fully-completed scenario costs nothing in resume mode.
        def _ensure_loaded():
            nonlocal df_old, df_new, df_wide
            if df_old is None:
                df_old = load_scenario_csv(DUMP_OLD, scenario, "nopattern")
            if df_new is None:
                df_new = load_scenario_csv(DUMP_NEW, scenario, "nopattern_noefficacy")
            if df_wide is None:
                df_wide = load_scenario_csv(DUMP_WIDE, scenario, "pattern")
            return df_old is not None and df_new is not None and df_wide is not None

        # Get the variable list cheaply: in resume mode we only need it if at
        # least one figure is missing, so peek using a quick header read.
        if resume:
            # Use the new dump as the reference for which variables to plot;
            # if everything for this scenario already exists, skip the load.
            ref_df = load_scenario_csv(DUMP_NEW, scenario, "nopattern_noefficacy")
            if ref_df is None:
                continue
            ref_vars = sorted(ref_df["variable"].unique())
            pending = []
            for variable in ref_vars:
                slug = slugify(variable)
                out_path = FIG_COMPARE_BY_VAR / slug / f"{scenario}_{slug}.png"
                if not out_path.exists():
                    pending.append(variable)
            if not pending:
                print(f"  all figures present, skipping {scenario}")
                continue
            df_new = ref_df  # reuse

        if not _ensure_loaded():
            continue

        vars_old = set(df_old["variable"].unique())
        vars_new = set(df_new["variable"].unique())
        vars_wide = set(df_wide["variable"].unique())
        common_vars = sorted(vars_old & vars_new & vars_wide)
        only_old_vars = sorted(vars_old - vars_new - vars_wide)
        only_new_vars = sorted(vars_new - vars_old - vars_wide)
        only_wide_vars = sorted(vars_wide - vars_old - vars_new)
        if only_old_vars:
            print(f"  variables only in old dump: {only_old_vars}")
        if only_new_vars:
            print(f"  variables only in new dump: {only_new_vars}")
        if only_wide_vars:
            print(f"  variables only in new dump: {only_wide_vars}")

        for variable in common_vars:
            slug = slugify(variable)
            by_var_dir = FIG_COMPARE_BY_VAR / slug
            out_path = by_var_dir / f"{scenario}_{slug}.png"
            if resume and out_path.exists():
                continue

            old_data = extract_ensemble(df_old, variable)
            new_data = extract_ensemble(df_new, variable)
            wide_data = extract_ensemble(df_wide, variable)
            if old_data is None or new_data is None or wide_data is None:
                continue

            years_o, arr_o = old_data
            years_n, arr_n = new_data
            years_w, arr_w = wide_data
            p05_o, p50_o, p95_o = percentile_band(arr_o)
            p05_n, p50_n, p95_n = percentile_band(arr_n)
            p05_w, p50_w, p95_w = percentile_band(arr_w)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_band(ax, years_n, p05_n, p50_n, p95_n, color="C0", label="no pattern, no efficacy")
            plot_band(ax, years_o, p05_o, p50_o, p95_o, color="C1", label="no pattern")
            plot_band(ax, years_w, p05_w, p50_w, p95_w, color="C2", label="pattern and efficacy")
            ax.set_title(f"{scenario} — {variable}")
            ax.set_xlabel("Year")
            unit_series = df_new.loc[df_new["variable"] == variable, "unit"]
            unit = unit_series.iloc[0] if not unit_series.empty else ""
            ax.set_ylabel(f"{variable} ({unit})" if unit else variable)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            by_var_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=120)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 3: flat10 replication
# ---------------------------------------------------------------------------
def build_flat10_long_df(folder: Path, scenarios: list[str], label:str) -> pd.DataFrame:
    """Concatenate per-scenario CSVs into a single long-form DataFrame matching
    the schema used by flat10_trials/make_timeseries_plots.py (year columns
    start at index 7)."""
    frames = []
    for scen in scenarios:
        df = load_scenario_csv(folder, scen, label)
        if df is None:
            print(f"[flat10] missing scenario CSV: {scen}")
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("No flat10 scenario CSVs could be loaded.")
    return pd.concat(frames, ignore_index=True)


def _times_axis(n_year_cols: int) -> np.ndarray:
    """Match flat10_trials convention: np.arange(-100, n_year_cols-100)."""
    return np.arange(-100, n_year_cols - 100)


def flat10_timeseries_plots(dfs_by_source: dict) -> None:
    """Overlay flat10 ensembles from multiple dump folders on one figure per
    variable. ``dfs_by_source`` maps source label -> long-form DataFrame."""
    # Use the new dump (preferred) for the time axis length.
    ref_df = next(iter(dfs_by_source.values()))
    n_year_cols = ref_df.shape[1] - 7
    times = _times_axis(n_year_cols)

    scenarios = sorted(set().union(*(set(df["scenario"].unique()) for df in dfs_by_source.values())))
    available_vars = set().union(*(set(df["variable"].unique()) for df in dfs_by_source.values()))

    # Linestyle per source, colour per scenario
    linestyles = {label: ls for label, ls in zip(dfs_by_source, ["-", "--", ":", "-."])}

    for short_name, (variable, ylabel) in FLAT10_VARIABLES.items():
        if variable not in available_vars:
            print(f"[flat10] skipping '{variable}' (not present in any dump)")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, scen in enumerate(scenarios):
            colour = FLAT10_COLOURS[i % len(FLAT10_COLOURS)]
            for src_label, df in dfs_by_source.items():
                sub = df.loc[(df["scenario"] == scen) & (df["variable"] == variable)]
                if sub.empty:
                    continue
                arr = sub.iloc[:, 7:].to_numpy(dtype=float)
                src_n_cols = arr.shape[1]
                src_times = _times_axis(src_n_cols)
                p05, p50, p95 = percentile_band(arr)
                ls = linestyles.get(src_label, "-")
                ax.fill_between(src_times, p05, p95, color=colour, alpha=0.15, linewidth=0)
                ax.plot(src_times, p50, color=colour, lw=1.5, linestyle=ls,
                        label=f"{scen} [{src_label}]")
        ax.set_xlabel("time (years; 0 = 1850)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"flat10 ensemble: {variable}")
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_FLAT10 / f"{short_name}_timeseries.png", dpi=120)
        plt.close(fig)


def _slice_mean(df_var: pd.DataFrame, s: int, e: int) -> np.ndarray:
    """Mean over years[s:e] using the flat10_trials offset convention
    (column 7 + 100 + s ... 7 + 100 + e)."""
    return df_var.iloc[:, 7 + 100 + s : 7 + 100 + e].to_numpy(dtype=float).mean(axis=1)


def flat10_key_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate flat10_trials/calculate_metrics.py."""
    var_t = "Surface Air Temperature Change"

    df_flat10 = df.loc[(df["scenario"] == "esm-flat10") & (df["variable"] == var_t)].reset_index(drop=True)
    df_pic = df.loc[(df["scenario"] == "esm-piControl") & (df["variable"] == var_t)].reset_index(drop=True)
    df_zec = df.loc[(df["scenario"] == "esm-flat10-zec") & (df["variable"] == var_t)].reset_index(drop=True)
    df_cdr = df.loc[(df["scenario"] == "esm-flat10-cdr") & (df["variable"] == var_t)].reset_index(drop=True)

    if df_flat10.empty or df_pic.empty:
        raise RuntimeError("Need esm-flat10 and esm-piControl for flat10 metrics.")

    n = min(len(df_flat10), len(df_pic), len(df_zec), len(df_cdr))
    df_flat10 = df_flat10.iloc[:n]
    df_pic = df_pic.iloc[:n]
    df_zec = df_zec.iloc[:n]
    df_cdr = df_cdr.iloc[:n]

    runids = df_flat10["run_id"].to_numpy()

    tcre = _slice_mean(df_flat10, 90, 110) - _slice_mean(df_pic, 90, 110)
    zec50 = _slice_mean(df_zec, 140, 160) - tcre
    zec100 = _slice_mean(df_zec, 190, 210) - tcre
    zec200 = _slice_mean(df_zec, 290, 310) - tcre
    zec190 = _slice_mean(df_zec, 280, 300) - tcre
    tnz = _slice_mean(df_cdr, 140, 160) - _slice_mean(df_flat10, 115, 135)
    tr1000 = _slice_mean(df_cdr, 190, 210) - _slice_mean(df_flat10, 90, 110)
    tr0 = _slice_mean(df_cdr, 300, 320) - _slice_mean(df_pic, 300, 320)

    # tpw: 20-yr rolling mean argmax of (cdr - piControl), shifted by -250.
    tpw = np.zeros(n)
    cdr_arr = df_cdr.iloc[:, 7:].to_numpy(dtype=float)
    pic_arr = df_pic.iloc[:, 7:].to_numpy(dtype=float)
    diff = cdr_arr - pic_arr
    window = np.ones(20) / 20.0
    idx_axis = np.arange(diff.shape[1])
    rolled_idx = np.convolve(idx_axis, window, mode="valid")
    for i in range(n):
        rolled = np.convolve(diff[i], window, mode="valid")
        tpw[i] = rolled_idx[int(np.argmax(rolled))] - 250

    metrics = pd.DataFrame(
        {
            "tcre": tcre,
            "zec50": zec50,
            "zec100": zec100,
            "zec200": zec200,
            "zec190": zec190,
            "tnz": tnz,
            "tr1000": tr1000,
            "tr0": tr0,
            "tpw": tpw,
        },
        index=runids,
    )
    metrics.index.name = "run_id"
    return metrics


def flat10_pairplot_histograms(metrics: pd.DataFrame, suffix: str = "") -> None:
    snsplot = sns.pairplot(
        metrics.drop(labels=["zec200", "zec190"], axis=1),
        corner=True,
        plot_kws={"alpha": 0.5},
        height=1,
    )
    snsplot.fig.savefig(FIG_FLAT10 / f"pairplot_ciceroscm{suffix}.png", dpi=120)
    plt.close(snsplot.fig)


def flat10_combined_pairplot(metrics_by_source: dict) -> None:
    """Render a single two-panel figure with one pairplot per source.

    Reads the per-source pairplot PNGs already produced by
    ``flat10_pairplot_histograms`` and pastes them side by side.
    """
    if len(metrics_by_source) < 2:
        return
    pngs = [(label, FIG_FLAT10 / f"pairplot_ciceroscm_{label}.png")
            for label in metrics_by_source]
    pngs = [(l, p) for l, p in pngs if p.exists()]
    if len(pngs) < 2:
        return

    fig, axes = plt.subplots(1, len(pngs), figsize=(7 * len(pngs), 7))
    if len(pngs) == 1:
        axes = [axes]
    for ax, (label, png) in zip(axes, pngs):
        ax.imshow(plt.imread(png))
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_FLAT10 / "pairplot_ciceroscm_combined.png", dpi=150)
    plt.close(fig)


def flat10_combined_histograms(metrics_by_source: dict) -> None:
    """Overlay histograms of key metrics across multiple sources."""
    if not metrics_by_source:
        return
    colours = {label: c for label, c in zip(metrics_by_source, ["C0", "C1", "C2", "C3"])}

    cols = ["tcre", "zec50", "zec100"]
    for col in cols:
        fig, ax = plt.subplots()
        for label, metrics in metrics_by_source.items():
            ax.hist(
                metrics[col].to_numpy(),
                bins=20,
                histtype="step",
                fill=False,
                color=colours[label],
                label=label,
                linewidth=1.5,
            )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_FLAT10 / f"{col}_histogram.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots()
    for label, metrics in metrics_by_source.items():
        ax.hist(
            (metrics["tcre"] + metrics["zec50"]).to_numpy(),
            bins=20,
            histtype="step",
            fill=False,
            color=colours[label],
            label=label,
            linewidth=1.5,
        )
    ax.set_title("tcre + zec50")
    ax.set_xlabel("tcre + zec50")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_FLAT10 / "tcre_plus_zec50_histogram.png", dpi=120)
    plt.close(fig)


def flat10_gregory_plots(folders_by_source: dict) -> None:
    """Overlay Gregory plots for each abrupt scenario across multiple dumps."""
    colours = {label: c for label, c in zip(folders_by_source, ["C0", "C1", "C2", "C3"])}
    for scen in ABRUPT_SCENARIOS:
        fig, ax = plt.subplots(figsize=(6, 5))
        plotted = False
        for src_label, folder in folders_by_source.items():
            df = load_scenario_csv(folder, scen, src_label)
            if df is None:
                print(f"[flat10] gregory: missing {scen} in {src_label}")
                continue
            t = extract_ensemble(df, "Surface Air Temperature Change")
            h = extract_ensemble(df, "Heat Uptake")
            if t is None or h is None:
                continue
            _, t_arr = t
            _, h_arr = h
            n = min(t_arr.shape[0], h_arr.shape[0])
            colour = colours[src_label]
            for i in range(n):
                ax.plot(t_arr[i], h_arr[i], alpha=0.2, lw=0.5, color=colour)
            # Legend proxy
            ax.plot([], [], color=colour, lw=2, label=src_label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Surface air temperature change (K)")
        ax.set_ylabel("Heat uptake (ZJ / yr)")
        ax.set_title(f"Gregory plot {scen}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(FIG_FLAT10 / f"gregory_{scen}.png", dpi=120)
        plt.close(fig)


def make_flat10_replication() -> None:
    FIG_FLAT10.mkdir(parents=True, exist_ok=True)

    sources = {"nopattern_noefficacy": DUMP_NEW, "nopattern": DUMP_OLD, "pattern": DUMP_WIDE}
    dfs_by_source = {}
    for label, folder in sources.items():
        try:
            dfs_by_source[label] = build_flat10_long_df(folder, FLAT10_SCENARIOS, label)
        except RuntimeError as exc:
            print(f"[flat10] could not load {label}: {exc}")
    if not dfs_by_source:
        print("[flat10] no dumps available, aborting")
        return

    flat10_timeseries_plots(dfs_by_source)

    metrics_by_source = {}
    for label, df in dfs_by_source.items():
        try:
            metrics = flat10_key_metrics(df)
        except RuntimeError as exc:
            print(f"[flat10] metrics skipped for {label}: {exc}")
            continue
        suffix = f"_{label}"
        metrics_path = FIG_FLAT10 / f"flat10_ciceroscm_key_metrics{suffix}.csv"
        metrics.to_csv(metrics_path)
        print(f"[flat10] wrote {metrics_path}")
        print(metrics.describe())
        flat10_pairplot_histograms(metrics, suffix=suffix)
        metrics_by_source[label] = metrics

    flat10_combined_histograms(metrics_by_source)
    flat10_combined_pairplot(metrics_by_source)
    flat10_gregory_plots(sources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-comparisons", action="store_true")
    parser.add_argument("--skip-flat10", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip comparison figures that already exist on disk.",
    )
    args = parser.parse_args()

    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_comparisons:
        make_comparison_plots(resume=args.resume)
    if not args.skip_flat10:
        make_flat10_replication()


if __name__ == "__main__":
    main()

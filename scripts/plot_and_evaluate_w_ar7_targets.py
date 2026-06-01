"""
plot_and_evaluate_w_ar7_targets.py

Evaluates the CICERO-SCM ensemble against AR7 WG1 Ch5 calibration targets.

Outputs
-------
plots/ar7_targets_timeseries.png
    5–95 % plume timeseries for GMST, EEI, CO₂ concentration, Aerosol ERF,
    TCR, and TCRE with calibration constraints overlaid.

plots/ar7_targets_composite.png
    Composite panel comparing ensemble scalar distributions against the
    calibration target 5th / 50th / 95th ranges for all 7 diagnostics.
"""
import sys
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TARGETS_CSV = (
    ROOT / "data"
    / "Calibration_targets_for_AR7_WG1_Ch5_emulators_Calibration_targets.csv"
)
OUTDIR    = Path(__file__).resolve().parent / "out_file_dump_patternv1"
PLOT_DIR  = Path(__file__).resolve().parent / "plots"
DATE_STAMP = "20260520"

# ── variable / experiment configuration ──────────────────────────────────────
# Variable names in model output are assumed to match the "Variable" column in
# the targets CSV.  ECS is the exception: it is computed via Gregory regression
# from abrupt-4xCO2 using the variable names below (adjust if needed).
TEMP_VAR_ABRUPT4X = "Surface Air Ocean Blended Temperature Change"   # temperature variable in abrupt-4xCO2
OHC_VAR_ABRUPT4X  = "Heat Content|Ocean"    # cumulative energy/OHC variable in abrupt-4xCO2


# Maps CSV "Variable" names → actual variable names in the model output files.
target_variable_renaming: dict[str, str] = {
    "ECS":              "Surface Air Ocean Blended Temperature Change",
    "TCR":              "Surface Air Ocean Blended Temperature Change",
    "TCRE":              "Surface Air Ocean Blended Temperature Change",
    "GMST":              "Surface Air Ocean Blended Temperature Change",
    "EEI":               "Heat Content|Ocean",
    "CO2 concentration": "Atmospheric Concentrations|CO2",
    "Aerosol ERF":       "Effective Radiative Forcing|Anthropogenic|Aerosol",
}
# CSV Variable → experiment filename prefix.
# Historical runs map to esm-allGHG-hist.
EXPERIMENT_FOR: dict[str, str] = {
    "ECS":               "abrupt-4xCO2",
    "TCR":               "1pctCO2",
    "TCRE":              "esm-flat10",
    "GMST":              "esm-allGHG-hist",
    "EEI":               "esm-allGHG-hist",
    "Aerosol ERF":       "esm-allGHG-hist",
    "CO2 concentration": "esm-allGHG-scen7-M",
}

# Variables for which a timeseries plume is plotted (in addition to a scalar).
TIMESERIES_VARS = ["GMST", "EEI", "CO2 concentration", "Aerosol ERF", "TCR", "TCRE"]

# The first 100 years of idealised runs (abrupt-4xCO2, 1pctCO2, esm-flat10) are
# a background spinup; the real forcing perturbation starts at this calendar year.
IDEALIZED_START_YEAR = 1850


# ── data helpers ──────────────────────────────────────────────────────────────
def load_experiment(exp_name: str) -> pd.DataFrame:
    """Load output CSV for *exp_name*."""
    test_name = OUTDIR / f"{exp_name}_rcmip_ciceroscm_{DATE_STAMP}.csv"
    if os.path.exists(test_name):
        return pd.read_csv(test_name, index_col=0)
    
    if len(glob.glob(str(OUTDIR / f"{exp_name}_rcmip_ciceroscm_*.csv"))) > 0:
        fname = glob.glob(str(OUTDIR / f"{exp_name}_rcmip_ciceroscm_*.csv"))[0]
        return pd.read_csv(
            fname,
            index_col=0,
    )
    if len(glob.glob(str(OUTDIR / f"{exp_name}_*.csv"))) > 0:
        fname = glob.glob(str(OUTDIR / f"{exp_name}_*.csv"))[0]
        return pd.read_csv(
            fname,
            index_col=0,
    )


def get_timeseries(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Return a (ensemble_member × year) DataFrame for *variable*.
    Year column labels are cast to integers.
    """
    rows = df[df["variable"] == variable]
    year_cols = [c for c in df.columns if str(c).isdigit()]
    if "ensemble_member" in rows.columns:
        ts = rows.set_index("ensemble_member")[year_cols]
    elif "run_id" in rows.columns:
        ts = rows.set_index("run_id")[year_cols]
    else:
        raise ValueError(f"Expected 'ensemble_member' or 'run_id' column in DataFrame for variable '{variable}'")
    ts.columns = ts.columns.astype(int)
    return ts.apply(pd.to_numeric, errors="coerce")


def years_between(ts: pd.DataFrame, y0: int, y1: int) -> list[int]:
    """Sorted list of integer column years in [y0, y1]."""
    return sorted(y for y in ts.columns if y0 <= y <= y1)


# ── ECS helper ────────────────────────────────────────────────────────────────
def compute_ecs_gregory(
    df_abrupt4x: pd.DataFrame,
    temp_var: str = TEMP_VAR_ABRUPT4X,
    ohc_var: str  = OHC_VAR_ABRUPT4X,
    n_years: int  = 150,
) -> pd.Series:
    """
    Estimate ECS (for 2 × CO₂) per ensemble member using the Gregory (2004)
    regression method on the first *n_years* of an abrupt-4 × CO₂ simulation.

    The net TOA energy imbalance N is approximated by the annual change in the
    cumulative EEI / OHC variable::

        N_t ≈ EEI_t − EEI_{t−1}

    The surface temperature anomaly ΔT is computed relative to the first year.
    A linear fit  N = F₄ₓ + β · ΔT  (β < 0) gives::

        ECS₄ₓ = −F₄ₓ / β   (equilibrium warming for 4 × CO₂)
        ECS    = ECS₄ₓ / 2   (one CO₂ doubling)
    """
    temp_ts = get_timeseries(df_abrupt4x, temp_var)
    ohc_ts  = get_timeseries(df_abrupt4x, ohc_var)

    # Skip the spinup: take n_years starting from IDEALIZED_START_YEAR.
    all_years  = sorted(temp_ts.columns)
    start_idx  = all_years.index(IDEALIZED_START_YEAR) if IDEALIZED_START_YEAR in all_years \
                 else next(i for i, y in enumerate(all_years) if y >= IDEALIZED_START_YEAR)
    years      = all_years[start_idx : start_idx + n_years]
    temp       = temp_ts[years].values          # (n_members, n_years)
    ohc        = ohc_ts[years].values

    delta_t   = temp - temp[:, [0]]             # anomaly relative to IDEALIZED_START_YEAR
    N         = np.diff(ohc, axis=1)            # annual OHC change ≈ TOA imbalance
    delta_t_N = delta_t[:, 1:]                  # aligned with N (years 2 … n_years)

    ecs_vals = []
    for i in range(len(temp_ts)):
        dT, n = delta_t_N[i], N[i]
        mask  = np.isfinite(dT) & np.isfinite(n)
        if mask.sum() < 10:
            ecs_vals.append(np.nan)
            continue
        coeffs = np.polyfit(dT[mask], n[mask], 1)   # [slope, intercept]
        slope, intercept = coeffs
        if slope >= 0:
            ecs_vals.append(np.nan)     # unphysical feedback
            continue
        ecs_vals.append(-intercept / slope / 2.0)

    return pd.Series(ecs_vals, index=temp_ts.index, name="ECS")


# ── diagnostic computation ────────────────────────────────────────────────────
def compute_all_diagnostics(targets: pd.DataFrame) -> dict:
    """
    Compute per-ensemble-member scalars (and timeseries where applicable) for
    every calibration target row.

    Returns
    -------
    dict mapping variable name to::

        {"scalars":    pd.Series,
         "timeseries": pd.DataFrame | None,
         "target_row": pd.Series}
    """
    _exp_cache: dict[str, pd.DataFrame] = {}

    def get_df(exp: str) -> pd.DataFrame:
        if exp not in _exp_cache:
            _exp_cache[exp] = load_experiment(exp)
        return _exp_cache[exp]

    results: dict = {}

    for _, row in targets.iterrows():
        var        = row["Variable"]
        output_var = target_variable_renaming.get(var, var)   # CSV name → output name

        if var not in EXPERIMENT_FOR:
            continue

        df         = get_df(EXPERIMENT_FOR[var])
        timeseries = None

        if var == "ECS":
            scalars = compute_ecs_gregory(df)

        elif var == "TCR":
            ts      = get_timeseries(df, output_var)
            # Anomaly relative to the start of the real experiment (post-spinup)
            ts_anom = ts.subtract(ts[IDEALIZED_START_YEAR], axis=0)
            yr70    = IDEALIZED_START_YEAR + 69   # year 70 of real experiment
            if yr70 not in ts_anom.columns:
                yr70 = min(ts_anom.columns, key=lambda y: abs(y - yr70))
            scalars    = ts_anom[yr70]
            timeseries = ts_anom

        elif var == "TCRE":
            ts      = get_timeseries(df, output_var)
            # Anomaly relative to the start of the real experiment (post-spinup)
            ts_anom = ts.subtract(ts[IDEALIZED_START_YEAR], axis=0)
            yr100   = IDEALIZED_START_YEAR + 99   # year 100 of real experiment
            if yr100 not in ts_anom.columns:
                yr100 = min(ts_anom.columns, key=lambda y: abs(y - yr100))
            scalars    = ts_anom[yr100]
            timeseries = ts_anom

        elif var == "GMST":
            ts       = get_timeseries(df, output_var)
            ref_cols = years_between(ts, 1850, 1900)
            ref      = ts[ref_cols].mean(axis=1)
            ts_anom  = ts.subtract(ref, axis=0)         # anomaly vs 1850–1900
            per_cols = years_between(ts, 2004, 2023)
            scalars    = ts_anom[per_cols].mean(axis=1)
            timeseries = ts_anom

        elif var == "EEI":
            ts         = get_timeseries(df, output_var)
            ts_rel     = ts.subtract(ts[1971], axis=0)  # relative to 1971
            scalars    = ts_rel[2020]
            timeseries = ts_rel

        elif var == "Aerosol ERF":
            ts       = get_timeseries(df, output_var)
            print(ts)
            sys.exit(1)
            ts_anom  = ts.subtract(ts[1750], axis=0)    # anomaly vs 1750
            per_cols = years_between(ts, 2005, 2014)
            scalars    = ts_anom[per_cols].mean(axis=1)
            timeseries = ts_anom

        elif var == "CO2 concentration":
            ts = get_timeseries(df, output_var)
            if 2025 not in ts.columns:
                # fall back to the historical run if scenario doesn't cover 2025
                ts = get_timeseries(get_df("esm-allGHG-hist"), output_var)
            yr       = min(ts.columns, key=lambda y: abs(y - 2025))
            scalars    = ts[yr]
            timeseries = ts

        else:
            continue

        results[var] = {
            "scalars":    scalars,
            "timeseries": timeseries,
            "target_row": row,
        }

    return results


# ── plot: timeseries plumes ───────────────────────────────────────────────────
def plot_timeseries_panels(results: dict) -> plt.Figure:
    """
    Up to 3 × 2 panel figure: 5–95 % shaded plume + median for each timeseries
    variable, with AR7 calibration constraints overlaid.
    """
    ts_vars = [v for v in TIMESERIES_VARS if v in results]
    ncols = 2
    nrows = (len(ts_vars) + ncols - 1) // ncols
    print(results.keys())
    print(nrows, ncols, ts_vars)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(13, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    for ax, var in zip(axes, ts_vars):
        info          = results[var]
        ts            = info["timeseries"]
        row           = info["target_row"]
        lo, mid, hi   = float(row["5th"]), float(row["50th"]), float(row["95th"])
        years         = sorted(ts.columns)

        p05 = ts[years].quantile(0.05)
        p50 = ts[years].quantile(0.50)
        p95 = ts[years].quantile(0.95)

        ax.fill_between(years, p05, p95, alpha=0.3, color="steelblue", label="Model 5–95 %")
        ax.plot(years, p50, color="steelblue", lw=1.5, label="Model median")

        _overlay_timeseries_constraint(ax, var, ts, lo, mid, hi)

        ax.set_title(f"{var}  [{row['Unit']}]", fontsize=10)
        ax.set_xlabel("Year")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    for ax in axes[len(ts_vars):]:
        ax.set_visible(False)

    fig.suptitle(
        "CICERO-SCM ensemble – timeseries plumes vs AR7 calibration targets",
        fontsize=12,
    )
    return fig


def _overlay_timeseries_constraint(ax, var, ts, lo, mid, hi):
    """Overlay calibration constraint markers appropriate to each variable."""
    orange = "darkorange"
    green  = dict(alpha=0.10, color="limegreen", zorder=2)

    if var == "GMST":
        ax.axhspan(lo, hi, alpha=0.20, color=orange, label="Target 5–95 %")
        ax.axhline(mid, color=orange, lw=1.5, ls="--", label="Target 50th")
        ax.axvspan(2004, 2023, **green, label="Constraint period")

    elif var == "EEI":
        ax.errorbar(
            2020, mid, yerr=[[mid - lo], [hi - mid]],
            fmt="D", color=orange, ms=7, capsize=5, zorder=5,
            label="Target 5th–95th",
        )
        ax.axvspan(1971, 2020, **green, label="Constraint period")

    elif var == "CO2 concentration":
        ax.errorbar(
            2025, mid, yerr=[[mid - lo], [hi - mid]],
            fmt="D", color=orange, ms=7, capsize=5, zorder=5,
            label="Target 2025",
        )
        ax.axvline(2025, color="limegreen", ls=":", lw=1.2, label="Target year")

    elif var == "Aerosol ERF":
        ax.axhspan(lo, hi, alpha=0.20, color=orange, label="Target 5–95 %")
        ax.axhline(mid, color=orange, lw=1.5, ls="--", label="Target 50th")
        ax.axvspan(2005, 2014, **green, label="Constraint period")

    elif var == "TCR":
        yr70 = IDEALIZED_START_YEAR + 69
        ax.errorbar(
            yr70, mid, yerr=[[mid - lo], [hi - mid]],
            fmt="D", color=orange, ms=7, capsize=5, zorder=5,
            label="Target at year 70",
        )
        ax.axvline(yr70, color="limegreen", ls=":", lw=1.2, label=f"Year 70 ({yr70})")

    elif var == "TCRE":
        yr100 = IDEALIZED_START_YEAR + 99
        ax.errorbar(
            yr100, mid, yerr=[[mid - lo], [hi - mid]],
            fmt="D", color=orange, ms=7, capsize=5, zorder=5,
            label="Target at year 100",
        )
        ax.axvline(yr100, color="limegreen", ls=":", lw=1.2, label=f"Year 100 ({yr100})")


# ── plot: composite scalar summary ────────────────────────────────────────────
def plot_composite_scalars(results: dict) -> plt.Figure:
    """
    One subplot per diagnostic showing a side-by-side range comparison:
      • Blue bar  – model ensemble 5th–95th percentile with 50th marked
      • Orange bar – AR7 calibration target 5th–95th percentile with 50th marked

    The bar width is purely cosmetic; only the y-extent (value range) matters.
    """
    BAR_W  = 0.35   # half-width of each range bar in x-axis units
    X_MOD  = 0.0    # x-position of model bar
    X_TGT  = 1.0    # x-position of target bar

    var_list = list(results.keys())
    ncols    = 4
    nrows    = (len(var_list) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    for ax, var in zip(axes, var_list):
        info          = results[var]
        scalars       = info["scalars"].dropna().values
        row           = info["target_row"]
        lo, mid, hi   = float(row["5th"]), float(row["50th"]), float(row["95th"])
        unit          = row["Unit"]

        # ── model range bar ──────────────────────────────────────────────────
        if len(scalars) >= 2:
            q05, q50, q95 = np.nanpercentile(scalars, [5, 50, 95])
            ax.bar(X_MOD, q95 - q05, bottom=q05, width=BAR_W * 2,
                   color="steelblue", alpha=0.5, zorder=2)
            ax.hlines(q50, X_MOD - BAR_W, X_MOD + BAR_W,
                      colors="steelblue", lw=2.5, zorder=3)
        elif len(scalars) == 1:
            q05 = q50 = q95 = scalars[0]
            ax.hlines(q50, X_MOD - BAR_W, X_MOD + BAR_W,
                      colors="steelblue", lw=2.5, zorder=3)
        else:
            q05 = q50 = q95 = np.nan

        # ── target range bar ─────────────────────────────────────────────────
        ax.bar(X_TGT, hi - lo, bottom=lo, width=BAR_W * 2,
               color="darkorange", alpha=0.45, zorder=2)
        ax.hlines(mid, X_TGT - BAR_W, X_TGT + BAR_W,
                  colors="darkorange", lw=2.5, zorder=3)

        # ── annotation ───────────────────────────────────────────────────────
        ax.text(
            X_MOD, q05 - (hi - lo) * 0.03,
            f"[{q05:.3g}, {q50:.3g}, {q95:.3g}]",
            ha="center", va="top", fontsize=7, color="steelblue",
        )
        ax.text(
            X_TGT, lo - (hi - lo) * 0.03,
            f"[{lo:.3g}, {mid:.3g}, {hi:.3g}]",
            ha="center", va="top", fontsize=7, color="darkorange",
        )

        ax.set_title(f"{var}\n[{unit}]", fontsize=9)
        ax.set_xticks([X_MOD, X_TGT])
        ax.set_xticklabels(["Model", "Target"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="y", alpha=0.3)

    # shared legend
    legend_handles = [
        mpatches.Patch(facecolor="steelblue",   alpha=0.5,  label="Model 5–95 %"),
        mpatches.Patch(facecolor="darkorange",  alpha=0.45, label="Target 5–95 %"),
        plt.Line2D([0], [0], color="steelblue",  lw=2.5, label="Model 50th"),
        plt.Line2D([0], [0], color="darkorange", lw=2.5, label="Target 50th"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )

    for ax in axes[len(var_list):]:
        ax.set_visible(False)

    fig.suptitle(
        "CICERO-SCM ensemble scalars vs AR7 WG1 Ch5 calibration targets",
        fontsize=13,
    )
    return fig



# ── inspection helper ─────────────────────────────────────────────────────────
def list_variables(*exp_names: str) -> None:
    """Print all unique variable names (and units) in one or more experiment files."""
    if not exp_names:
        exp_names = tuple(EXPERIMENT_FOR.values())
    for exp_name in dict.fromkeys(exp_names):   # deduplicated, order preserved
        df = load_experiment(exp_name)
        summary = (
            df[["variable", "unit"]]
            .drop_duplicates()
            .sort_values("variable")
            .reset_index(drop=True)
        )
        print(f"\nVariables in '{exp_name}':\n")
        print(summary.to_string(index=False))
        print()

def get_plotname_suffix()-> str:
    """Return a string suffix for plot filenames based on the experiments used."""
    if OUTDIR is None:
        return "_unknown"
    outdir_name = str(OUTDIR.name).split("/")[-1]
    if not outdir_name.startswith("out_file_dump"):
        return f"_{outdir_name}"
    if outdir_name == "out_file_dump":
        return ""
    return outdir_name[len('out_file_dump'):]

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    PLOT_DIR.mkdir(exist_ok=True)
    targets = pd.read_csv(TARGETS_CSV)

    print("Computing diagnostics …")
    results = compute_all_diagnostics(targets)

    suffix = get_plotname_suffix()

    print("Plotting timeseries plumes …")
    fig_ts  = plot_timeseries_panels(results)
    out_ts  = PLOT_DIR / f"ar7_targets_timeseries{suffix}.png"
    fig_ts.savefig(out_ts, dpi=150)
    plt.close(fig_ts)
    print(f"  → {out_ts}")

    print("Plotting composite scalar summary …")
    fig_comp = plot_composite_scalars(results)
    out_comp = PLOT_DIR / f"ar7_targets_composite{suffix}.png"
    fig_comp.savefig(out_comp, dpi=150)
    plt.close(fig_comp)
    print(f"  → {out_comp}")


if __name__ == "__main__":
    #list_variables()   # inspect variable names; switch to main() when ready
    main()

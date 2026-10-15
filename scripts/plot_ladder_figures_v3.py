#!/usr/bin/env python
"""Generate the cumulative-ladder comparison figures for the paper.

Self-contained companion to ``extract_ladder_metrics.py`` (it reuses that
module's loaders and metric functions). Reads Marit's dumps read-only by
absolute path and writes PNGs into ``analysis/figures/`` in the paper repo, so
the manuscript can ``\\includegraphics`` them in place of the placeholder
floats. Can be copied verbatim into ``cscm-calibrate/scripts`` once the
remaining (+PDO) dumps exist; just extend the ``LADDER`` registry in
``extract_ladder_metrics.py``.

Produces:
    ladder_sensitivity_marginals.png  -> fig:posterior_marginals (ECS, TCR)
    ladder_projection_violins.png     -> fig:projection_spread
    ladder_flat10_metrics.png         -> fig:flat10_metrics (TCRE, ZEC50)
    ladder_seferian3.png              -> 1pctCO2 carbon-cycle ladder overlay
    ladder_gregory_ecs_plumes.png     -> rolling Gregory ECS plumes for abrupt runs

Usage:
    python plot_ladder_figures.py
"""
from __future__ import annotations
import glob
import sys
import plot_sanity_checks

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib.lines import Line2D
from statsmodels.regression.rolling import RollingOLS


import extract_ladder_metrics as elm
from plot_carbon_cycle_from_dumps import (
    _1PCT_END_IDX,
    _1PCT_START_IDX,
    _import_look_at_data,
    _look_at_data_cwd,
    SEFERIAN_VARS as CARBON_SEFERIAN_VARS,
    extract_or_derive as carbon_extract_or_derive,
    percentile_band as carbon_percentile_band,
)
from plot_various_from_experiments import make_flat10_key_metrics

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Consistent colour per rung across all figures.
DEFAULT_COLOURS = {
    "0 baseline (v1)": "#444444",
    "1 +efficacy": "#1f77b4",
    "2 +pattern": "#d62728",
    "3 +PDO": "#9467bd",
    "4a IPCC+pattern": "#2ca02c",
    "4b IPCC+pat+PDO": "#ff7f0e",
}
COLOURS = DEFAULT_COLOURS.copy()

ABRUPT_PANELS = [
    ("abrupt-4xCO2", "abrupt-4xCO2", 1),
    ("abrupt-2xCO2", "abrupt-2xCO2", 2),
    ("abrupt-0.5xCO2", "abrupt-0p5xCO2", -2),
]
GREGORY_N_YEARS = np.arange(50, 301, 5)

# scen7 ScenarioMIP categories and the blended-temperature / local-lambda panel.
SCEN7_CATEGORIES = ["L", "M", "VL", "LN", "ML", "H", "HL"]
SCEN7_SCENARIO_TEMPLATE = "esm-allGHG-scen7-{cat}"
SCEN7_END_YEAR = 2100
BLENDED_TEMP_VAR = elm.TEMP_VAR  # "Surface Air Ocean Blended Temperature Change"
RIB_VAR = "Heat Uptake"  # radiative imbalance proxy (RIB_glob in raw output)
LOCAL_LAMBDA_WINDOW = 30
PDO_RUNG_LABELS = {"3 +PDO", "4b IPCC+pat+PDO"}
NO_PDO_DISPLAY_LABELS = {
    "0 baseline (v1)": "baseline",
    "1 +efficacy": "efficacy",
    "2 +pattern": "aerosol",
    "4a IPCC+pattern": "aerosol ECS constrained",
}
ROLLING_WINDOW = 5


def active_ladder(*, ladder=None, include_pdo: bool = True):
    """Return the ladder entries to include in plots."""
    if ladder is not None:
        return list(ladder)
    if include_pdo:
        return elm.LADDER
    return [entry for entry in elm.LADDER if entry[0] not in PDO_RUNG_LABELS]


def display_label(
    label: str,
    *,
    include_pdo: bool = True,
    label_overrides: dict[str, str] | None = None,
) -> str:
    """Return the label used in plot annotations."""
    if label_overrides and label in label_overrides:
        return label_overrides[label]
    if include_pdo:
        return label
    return NO_PDO_DISPLAY_LABELS.get(label, label)


def _normalise_output_suffix(output_suffix: str) -> str:
    if not output_suffix:
        return ""
    return output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"


def figure_path(filename: str, *, include_pdo: bool = False, output_suffix: str = "") -> Path:
    """Return the output path for a figure, using an explicit suffix when given."""
    if output_suffix:
        stem, suffix = filename.rsplit(".", 1)
        return FIG_DIR / f"{stem}{_normalise_output_suffix(output_suffix)}.{suffix}"
    if not include_pdo:
        return FIG_DIR / filename
    stem, suffix = filename.rsplit(".", 1)
    return FIG_DIR / f"{stem}_full_ladder.{suffix}"


def build_rung_colours(
    rungs,
    *,
    colour_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a deterministic colour per rung, filling gaps from tab20."""
    palette = [matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]
    assigned = {}
    used = set()
    for label in rungs:
        colour = None
        if colour_overrides is not None:
            colour = colour_overrides.get(label)
        if colour is None:
            colour = DEFAULT_COLOURS.get(label)
        if colour is not None:
            assigned[label] = colour
            used.add(colour)

    fallback_index = 0
    for label in rungs:
        if label in assigned:
            continue
        while fallback_index < len(palette) and palette[fallback_index] in used:
            fallback_index += 1
        if fallback_index >= len(palette):
            fallback_index = 0
        assigned[label] = palette[fallback_index]
        used.add(palette[fallback_index])
        fallback_index += 1
    return assigned


def load_rung_scenario(
    scenario: str, *, ladder=None, include_pdo: bool = True
) -> dict[str, pd.DataFrame]:
    """Return available scenario outputs for every ladder rung."""
    out = {}
    for label, dump, _, _ in active_ladder(ladder=ladder, include_pdo=include_pdo):
        df = elm.load_scenario(elm.DUMP_ROOT / dump, scenario)
        if df is not None:
            out[label] = df
    return out


def _plot_band(ax, years, p05, p50, p95, *, color: str, label: str | None) -> None:
    """Plot a shaded 5th-95th percentile band and its median."""
    ax.fill_between(years, p05, p95, color=color, alpha=0.20, linewidth=0)
    ax.plot(years, p50, color=color, lw=1.6, label=label)


def _set_panel_label(ax, label: str | None) -> None:
    """Add a left-aligned panel label when requested."""
    if label:
        ax.set_title(label, loc="left", fontweight="bold")


def draw_sensitivity_marginals(
    axes,
    data,
    rungs,
    rung_display_labels,
    rung_colours,
    targets,
    *,
    panel_labels: tuple[str, str] | None = None,
) -> None:
    """Draw the ECS and TCR posterior marginals into the provided axes."""
    for idx, (var, ax) in enumerate(zip(("ECS", "TCR"), axes)):
        for r in rungs:
            v = data[r].get(var)
            if v is None:
                continue
            ax.hist(
                v,
                bins=40,
                range=(0, 10 if var == "ECS" else 5),
                histtype="step",
                density=True,
                lw=2,
                color=rung_colours[r],
                label=rung_display_labels[r],
            )
            ax.axvline(np.median(v), color=rung_colours[r], ls=":", lw=1)
        if var in targets:
            t = targets[var]
            ax.axvline(t[0], color="k", ls="--", lw=1, label="IPCC median target")
            ax.axvline(t[1], color="k", ls=":", lw=1, label="IPCC 5-95% range")
            ax.axvline(t[2], color="k", ls=":", lw=1)
        ax.set_xlabel(f"{var} (K)")
        ax.set_ylabel("posterior density")
        ax.set_title(var)
        if panel_labels is not None:
            _set_panel_label(ax, panel_labels[idx])
    axes[0].legend(fontsize=8, frameon=False)


def draw_projection_violins(
    axes,
    data,
    rungs,
    rung_display_labels,
    rung_colours,
    *,
    panel_labels: tuple[str, str] | None = None,
) -> None:
    """Draw the end-of-century warming boxplots into the provided axes."""
    for idx, ((tag, title), ax) in enumerate(
        zip(
            [("scen7H", "scen7-H (high)"), ("scen7VL", "scen7-VL (very low)")],
            axes,
        )
    ):
        series, labels, colours = [], [], []
        for r in rungs:
            v = data[r].get(tag)
            if v is not None:
                series.append(v)
                labels.append(rung_display_labels[r])
                colours.append(rung_colours[r])
        if not series:
            ax.set_ylabel(r"$\Delta T_{2081-2100}$ vs 1850-1900 (K)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            if panel_labels is not None:
                _set_panel_label(ax, panel_labels[idx])
            continue
        bp = ax.boxplot(
            series,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="k"),
        )
        for box, c in zip(bp["boxes"], colours):
            box.set_facecolor(c)
            box.set_alpha(0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(r"$\Delta T_{2081-2100}$ vs 1850-1900 (K)")
        ax.set_title(title)
        if panel_labels is not None:
            _set_panel_label(ax, panel_labels[idx])


def draw_scen7_temp_lambda_row(
    ax_temp,
    ax_lam,
    cat: str,
    *,
    internal_lambda: bool = True,
    ladder=None,
    include_pdo: bool = True,
    rung_display_labels: dict[str, str] | None = None,
    rung_colours: dict[str, str] | None = None,
    panel_labels: tuple[str, str] | None = None,
    just_means: bool = False,
) -> bool:
    """Draw one scen7 temperature/lambda panel pair into supplied axes."""
    scenario = SCEN7_SCENARIO_TEMPLATE.format(cat=cat)
    sources = load_rung_scenario(scenario, ladder=ladder, include_pdo=include_pdo)
    if not sources:
        print(f"[scen7] no outputs found for {scenario}, skipping")
        return False

    for label, df in sources.items():
        colour = (rung_colours or {}).get(label, DEFAULT_COLOURS.get(label, "k"))
        plot_label = display_label(
            label,
            include_pdo=include_pdo,
            label_overrides=rung_display_labels,
        )

        temp_ts = elm.timeseries(df, BLENDED_TEMP_VAR)
        if temp_ts.empty:
            continue
        temp_years = [y for y in sorted(temp_ts.columns) if y <= SCEN7_END_YEAR]
        if temp_years:
            temp_arr = temp_ts[temp_years].to_numpy(dtype=float)
            p05, p50, p95 = np.nanpercentile(temp_arr, [5, 50, 95], axis=0)
            _plot_band(
                ax_temp,
                temp_years,
                p05,
                p50,
                p95,
                color=colour,
                label=plot_label,
            )
        if not internal_lambda:
            rib_ts = elm.timeseries(df, RIB_VAR)
            if rib_ts.empty:
                continue
            members = temp_ts.index.intersection(rib_ts.index)
            common_years = [
                y
                for y in sorted(set(temp_ts.columns) & set(rib_ts.columns))
                if y <= SCEN7_END_YEAR
            ]
            if len(members) == 0 or len(common_years) <= LOCAL_LAMBDA_WINDOW:
                continue
            t_arr = temp_ts.loc[members, common_years].to_numpy(dtype=float)
            r_arr = rib_ts.loc[members, common_years].to_numpy(dtype=float)
            lam = np.full_like(t_arr, np.nan, dtype=float)
            for k in range(t_arr.shape[0]):
                lam[k] = local_lambda(t_arr[k], r_arr[k], window=LOCAL_LAMBDA_WINDOW)
            l05, l50, l95 = np.nanpercentile(lam, [5, 50, 95], axis=0)
            _plot_band(
                ax_lam,
                common_years,
                l05,
                l50,
                l95,
                color=colour,
                label=plot_label,
            )
            ax_lam.set_ylabel(r"Local feedback dN/dT (W m$^{-2}$ K$^{-1}$)")
            ax_lam.set_title(f"scen7-{cat}: {LOCAL_LAMBDA_WINDOW}-yr local feedback")
        else:
            lam_ts = elm.timeseries(df, "Dynamic Feedback Lambda")
            if lam_ts.empty:
                continue
            lam_years = [y for y in sorted(lam_ts.columns) if y <= SCEN7_END_YEAR]
            if lam_years:
                lam_arr = -lam_ts[lam_years].to_numpy(dtype=float)
                l05, l50, l95 = np.nanpercentile(lam_arr, [5, 50, 95], axis=0)
                if label in ["0 baseline (v1)", "1 +efficacy"]:
                    ax_lam.plot(lam_years, l50, color=colour, lw=1.6)
                    ax_lam.plot(lam_years, l05, color=colour, ls="--", lw=0.5)
                    ax_lam.plot(lam_years, l95, color=colour, ls="--", lw=0.5)
                else:
                    #l05 = rolling_mean_timeseries(pd.Series(l05), window=ROLLING_WINDOW)
                    #l50 = rolling_mean_timeseries(pd.Series(l50), window=ROLLING_WINDOW)
                    #l95 = rolling_mean_timeseries(pd.Series(l95), window=ROLLING_WINDOW)
                    _plot_band(ax_lam, lam_years, l05, l50, l95, color=colour, label=None)
                ax_lam.set_ylabel(r"Dynamic lambda feedback (W m$^{-2}$ K$^{-1}$)")

    ax_lam.set_title(f"scen7-{cat}: internally diagnosed lambda feedback")
    read_in_and_overlay_temperature_timeseries(ax_temp)
    read_in_and_overlay_andrews_2022_local_lambda(ax_lam, just_means=just_means)
    read_in_and_overlay_van_loon_2026_local_lambda(ax_lam, just_means=just_means)
    lam_to_ECS = lambda x: -3.7 / x
    ECS_to_lam = lambda x: -3.7 / x
    ax_lam2 = ax_lam.secondary_yaxis("right", functions=(lam_to_ECS, ECS_to_lam))
    ax_lam2.set_ylabel("ECS implied by dynamic lambda (K)")

    ax_temp.set_xlim(left=1850, right=SCEN7_END_YEAR)
    ax_temp.set_xlabel("Year")
    ax_temp.set_ylabel("Blended surface temperature change (K)")
    ax_temp.set_title(f"scen7-{cat}: blended temperature")
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(fontsize=7, frameon=False)

    ax_lam.set_xlim(left=1850, right=SCEN7_END_YEAR)# - 50)
    ax_lam.set_ylim(-2.5, -0.5)
    ax_lam.set_xlabel("Year (centre of window)")
    ax_lam.grid(True, alpha=0.3)

    if panel_labels is not None:
        _set_panel_label(ax_temp, panel_labels[0])
        _set_panel_label(ax_lam, panel_labels[1])

    return True

def rolling_mean_timeseries(ts_data, window: int) -> pd.Series:
    """Return the rolling mean of a time series with the specified window size."""
    return ts_data.rolling(window=window, min_periods=1, center=True).mean()

def rolling_gregory_ecs(
    abrupt_data: pd.DataFrame,
    *,
    n_years_values: np.ndarray = GREGORY_N_YEARS,
) -> pd.DataFrame:
    """Return per-member Gregory ECS values for a sweep of regression lengths."""
    series_by_n_years = []
    valid_n_years = []
    for n_years in n_years_values:
        ecs = elm.ecs_gregory(abrupt_data, n_years=int(n_years))
        series_by_n_years.append(ecs.rename(int(n_years)))
        valid_n_years.append(int(n_years))

    if not series_by_n_years:
        return pd.DataFrame()

    out = pd.concat(series_by_n_years, axis=1)
    out.columns = valid_n_years
    return out


def per_member(rung_dump: str):
    """Return dict of per-member metric arrays for one rung's dump."""
    dump = elm.DUMP_ROOT / rung_dump
    out = {}
    a4 = elm.load_scenario(dump, "abrupt-4xCO2")
    if a4 is not None:
        out["ECS"] = elm.ecs_gregory(a4).dropna().values
    p1 = elm.load_scenario(dump, "1pctCO2")
    if p1 is not None:
        out["TCR"] = elm.transient(p1, 69).dropna().values
    for tag, scen in elm.PROJ_SCENARIOS.items():
        df = elm.load_scenario(dump, scen)
        if df is not None:
            out[tag] = elm.eoc_warming(df).dropna().values
    return out


def make_seferian3_ladder_figure(*, include_pdo: bool = True) -> None:
    """Overlay the 1pctCO2 Seferian-3 style diagnostics across ladder rungs."""
    ladder = active_ladder(include_pdo=include_pdo)
    rung_display_labels = {
        label: display_label(label, include_pdo=include_pdo) for label, _, _, _ in ladder
    }
    rung_colours = build_rung_colours(rung_display_labels)
    sources_1pct = load_rung_scenario("1pctCO2", ladder=ladder, include_pdo=include_pdo)
    if not sources_1pct:
        print("[seferian3] no 1pctCO2 outputs found, skipping")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    seferian_var_list = list(CARBON_SEFERIAN_VARS.items())
    lad = _import_look_at_data()
    external_legend_handles = []
    if lad is not None:
        external_legend_handles = [
            Line2D([0], [0], color="C0", lw=1.5, label="CMIP6 1pctCO2"),
            Line2D([0], [0], color="pink", lw=1.5, label="CMIP6 1pctCO2-bgc"),
        ]

    for i, (_short_name, (variable, ylabel, cmip6_var)) in enumerate(seferian_var_list):
        ax = axs[i // 2, i % 2]
        for label, df in sources_1pct.items():
            result = carbon_extract_or_derive(df, variable)
            if result is None:
                continue
            _years, arr = result
            if arr.shape[1] <= _1PCT_START_IDX:
                continue
            end_idx = min(_1PCT_END_IDX, arr.shape[1])
            arr_slice = arr[:, _1PCT_START_IDX:end_idx]
            if arr_slice.size == 0:
                continue
            model_years = np.arange(arr_slice.shape[1])
            p05, p50, p95 = carbon_percentile_band(arr_slice)

            _plot_band(
                ax,
                model_years,
                p05,
                p50,
                p95,
                color=rung_colours[label],
                label=rung_display_labels[label],
            )

        if lad is not None:
            try:
                with _look_at_data_cwd():
                    lad.add_cmip6_lines_to_plot("1pctCO2", cmip6_var, ax)
                with _look_at_data_cwd():
                    lad.add_cmip6_lines_to_plot("1pctCO2-bgc", cmip6_var, ax, cl="pink")
            except Exception as exc:
                print(f"[seferian3] warning: external {cmip6_var} overlay failed: {exc}")

        ax.set_xlim(0, 150)
        ax.set_xlabel("Model time (yr; 0 = 1850)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(i + 97)})", fontsize=11, loc="left")
        ax.grid(True, alpha=0.3)

    ax_scatter = axs[1, 1]
    for label, df in sources_1pct.items():
        h_res = carbon_extract_or_derive(df, "Ocean heat uptake")
        c_res = carbon_extract_or_derive(df, "Ocean carbon flux")
        if h_res is None or c_res is None:
            continue
        _yh, h_arr = h_res
        _yc, c_arr = c_res
        n_members = min(h_arr.shape[0], c_arr.shape[0])
        end_idx = min(_1PCT_END_IDX, h_arr.shape[1], c_arr.shape[1])
        if end_idx <= _1PCT_START_IDX:
            continue
        h_slice = h_arr[:n_members, _1PCT_START_IDX:end_idx]
        c_slice = c_arr[:n_members, _1PCT_START_IDX:end_idx]
        for j in range(n_members):
            ax_scatter.plot(
                h_slice[j],
                c_slice[j],
                color=rung_colours[label],
                alpha=0.05,
                lw=0.5,
            )
        ax_scatter.plot(
            np.nanmedian(h_slice, axis=0),
            np.nanmedian(c_slice, axis=0),
            color=rung_colours[label],
            lw=2.0,
            label=rung_display_labels[label],
        )

    if lad is not None:
        try:
            with _look_at_data_cwd():
                lad.add_cmip6_lines_combined_hfds_fgco2(ax_scatter)
            with _look_at_data_cwd():
                lad.add_cmip6_lines_combined_hfds_fgco2(
                    ax_scatter, exp="1pctCO2-bgc", cl="pink"
                )
        except Exception as exc:
            print(f"[seferian3] warning: external scatter overlay failed: {exc}")

    ax_scatter.set_xlabel(r"Heat uptake (ZJ yr$^{-1}$)")
    ax_scatter.set_ylabel(r"Ocean C uptake (Pg C yr$^{-1}$)")
    ax_scatter.set_title("d)", fontsize=11, loc="left")
    ax_scatter.grid(True, alpha=0.3)

    panel_handles, panel_labels = axs[0, 1].get_legend_handles_labels()
    axs[0, 1].legend(panel_handles + external_legend_handles, panel_labels + [h.get_label() for h in external_legend_handles], fontsize=7)

    scatter_handles, scatter_labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(scatter_handles + external_legend_handles, scatter_labels + [h.get_label() for h in external_legend_handles], fontsize=7)

    fig.tight_layout()
    fig.savefig(figure_path("ladder_seferian3.png", include_pdo=include_pdo), dpi=150)
    plt.close(fig)


def make_gregory_ecs_plume_figure(*, include_pdo: bool = True) -> None:
    """Plot rolling Gregory ECS plume diagnostics for the abrupt ladder runs."""
    ladder = active_ladder(include_pdo=include_pdo)
    rung_display_labels = {
        label: display_label(label, include_pdo=include_pdo) for label, _, _, _ in ladder
    }
    rung_colours = build_rung_colours(rung_display_labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (title, scenario, scaling) in zip(axes, ABRUPT_PANELS):
        sources = load_rung_scenario(scenario, ladder=ladder, include_pdo=include_pdo)
        for label, df in sources.items():
            ecs_by_member = rolling_gregory_ecs(df)
            if ecs_by_member.empty:
                continue
            x = ecs_by_member.columns.to_numpy(dtype=float)
            ecs_values = ecs_by_member.to_numpy(dtype=float) * scaling
            p05, p50, p95 = np.nanpercentile(ecs_values, [5, 50, 95], axis=0)
            ax.fill_between(
                x,
                p05,
                p95,
                color=rung_colours[label],
                alpha=0.20,
                linewidth=0,
            )
            ax.plot(
                x,
                p50,
                color=rung_colours[label],
                lw=2.0,
                label=rung_display_labels[label],
            )
        ax.set_title(title)
        ax.set_xlabel("Gregory regression length (yr)")
        ax.set_xlim(GREGORY_N_YEARS[0], GREGORY_N_YEARS[-1])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Gregory ECS (K)")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(
        figure_path("ladder_gregory_ecs_plumes.png", include_pdo=include_pdo), dpi=150
    )
    plt.close(fig)


def local_lambda(temp: np.ndarray, rib: np.ndarray, window: int = LOCAL_LAMBDA_WINDOW) -> np.ndarray:
    """OLS slope of RIB on dT in a centred window (Andrews et al. 2022).

    Mirrors ``pdo_sweep_plot.local_lambda``: returns an array the same length
    as ``temp`` with NaN where the centred window falls outside the series.
    """
    half = window // 2
    out = np.full_like(temp, np.nan, dtype=float)
    for t in range(half, len(temp) - (window - half)):
        x = temp[t - half : t - half + window]
        y = rib[t - half : t - half + window]
        xm = x - x.mean()
        denom = (xm * xm).sum()
        if denom > 0:
            out[t] = (xm * (y - y.mean())).sum() / denom
    return out

def read_in_and_overlay_andrews_2022_local_lambda(ax, just_means = True) -> None:
    """Overlay the local lambda diagnostics from Andrews et al. (2022) Fig. 3."""
    print("Reading and plotting Andrews et al. (2022) local lambda diagnostics")
    andrews_data_dir = Path("/div/no-backup-nac/users/masan/GRAFITE/amip-hadSST-piForcing/data")
    for source in ["amip", "hadSST"]:
        lamda_loc_mean = None
        lamda_loc_num = 0
        mean_years = None
        for file in glob.glob(f"{andrews_data_dir}/{source}-piForcing*.nc"):
            try:
                print(f"Reading and plotting Andrews et al. (2022) local lambda from {file}")
                ds = xr.open_dataset(file)
                time = ds["year"].values
                dt = ds["dt"].values
                dn = ds["dn"].values
                lambda_loc = local_lambda(dt, dn, window=LOCAL_LAMBDA_WINDOW)
                if lamda_loc_mean is None:
                    lamda_loc_mean = lambda_loc
                    mean_years = time
                else:
                    if not np.array_equal(mean_years, time):
                        # Find common years
                        common_years = np.intersect1d(mean_years, time)
                        # Get indices for common years in both arrays
                        idx_mean = np.isin(mean_years, common_years)
                        idx_time = np.isin(time, common_years)
                        # Compress to common years and add
                        lamda_loc_mean = lamda_loc_mean[idx_mean] + lambda_loc[idx_time]
                        mean_years = common_years
                    else:                  
                        lamda_loc_mean = lamda_loc_mean + lambda_loc


                lamda_loc_num = lamda_loc_num + 1
                if not just_means:
                    if np.any(lambda_loc> -0.5):
                        print(f"Warning: local lambda from {file} contains very small values, check data and windowing")
                        continue
                    lambda_loc = rolling_mean_timeseries(pd.Series(lambda_loc), window=ROLLING_WINDOW).values
                    ax.plot(time, lambda_loc, color = "grey", lw=0.5, alpha=0.5)
            except Exception as exc:
                print(f"Warning: failed to read or plot {file}: {exc}")
        if lamda_loc_num > 0:
            lamda_loc_mean = lamda_loc_mean / lamda_loc_num
            lamda_loc_mean = rolling_mean_timeseries(pd.Series(lamda_loc_mean), window=ROLLING_WINDOW).values
            ax.plot(mean_years, lamda_loc_mean, color = "black" if source == "amip" else "grey", lw=1.0, label=f"Andrews+2022 {source}")
            ax.legend(fontsize=7, frameon=False)

        else:
            print(f"Warning: no valid local lambda data found for source {source} in Andrews et al. (2022) diagnostics")

def van_loon_five_year_premean_regression(fpath, direct = True) -> list:
    ds = xr.open_dataset(fpath)
    time = ds["end_year"].values
    year = ds["year"].values
    if direct:
        print(ds.variables)
        if "l_true" not in ds.variables:
            lambda_data = ds["l"]
        else:
            lambda_data = ds["l_true"]    
        time = ds["end_year"].values
        if "end_year" not in lambda_data.coords:
            time = year
        if lambda_data.ndim > 2 and "member" in lambda_data.coords:
            lambda_data = lambda_data.mean(dim="member")
        return time, lambda_data
    if "R" in ds.variables:
        rdata = ds["R"].squeeze()
    elif "R_true" in ds.variables:
        rdata = ds["R_true"].squeeze()
    tdata = ds["T"].squeeze()
    lambd_data = np.zeros_like(tdata.values)
    print(rdata)
    print(tdata)
    print(time)
    print(rdata.shape)
    print(tdata.shape)
    if len(rdata.shape) > 2 or len(tdata.shape) > 2:
        raise ValueError("Unexpected data shape for R or T.")
    if len(rdata.shape) == 1:
        rdata_here = rolling_mean_timeseries(pd.Series(rdata), window=5).values
    if len(tdata.shape) == 1:
        tdata_here = rolling_mean_timeseries(pd.Series(tdata), window=5).values
        mod = RollingOLS(rdata_here, tdata_here, window=5)
        fit = mod.fit(params_only = True)
        lambd_data[ :] = fit.params[0]
    else:
        for i in range(tdata.shape[0]):   
            tdata_here = rolling_mean_timeseries(pd.Series(tdata[i, :]), window=5).values
            if len(rdata.shape) > 1:
                rdata_here = rolling_mean_timeseries(pd.Series(rdata[i, :]), window=5).values
            mod =RollingOLS(rdata_here, tdata_here, window=5)
            fit = mod.fit(params_only = True)
            print(rdata_here)
            print(tdata_here)
            print(fit.params)
            print(fit.params.shape)
            lambd_data[i, :] = fit.params[:,0]
    return year, xr.DataArray(lambd_data, coords=[("member", np.arange(lambd_data.shape[0])), ("year", year)])

    
def read_in_and_overlay_van_loon_2026_local_lambda(ax, just_means = True) -> None:
    """Overlay the local lambda diagnostics from van Loon et al. (2026) Fig. 2."""
    print("Reading and plotting van Loon et al. (2026) local lambda diagnostics")
    vanloon_data_dir = Path("/div/no-backup-nac/users/masan/GRAFITE/vanLoon_data/")
    for file in glob.glob(f"{vanloon_data_dir}/*_tas2Rg_amip_trend_random_*.nc"):
        label = file.split("_")[-1].replace(".nc", "")
        try:
            time, lambda_data = van_loon_five_year_premean_regression(file)
        except ValueError as e:
            print(f"Error processing file {file}: {e}")
            continue
        time, lambda_data = van_loon_five_year_premean_regression(file)
        print(lambda_data.shape)
        print(time)
        print(f"Reading and plotting van Loon et al. (2026) local lambda from {file}")
        if not just_means:
            for member in range(lambda_data.shape[0]):
                lambda_loc = lambda_data[member, :].values
                if len(lambda_loc.shape) > 1:
                    print("Flattening")
                    lambda_loc = lambda_loc.flatten()
                #lambda_loc = rolling_mean_timeseries(pd.Series(lambda_loc), window=ROLLING_WINDOW).values 
                ax.plot(time, lambda_loc, color = "grey", lw=0.5, alpha=0.5)       
        mean_lambda_loc = np.nanmean(lambda_data.values, axis=0)
        #print(mean_lambda_loc)
        #print(mean_lambda_loc.shape)
        replot_mean = True
        if len(mean_lambda_loc.shape) > 1:
            if mean_lambda_loc.shape[0] > 1:
                print("Warning: mean local lambda has more than one member, check data and windowing")
                # for member in range(mean_lambda_loc.shape[0]):
                #     print(f"Member {member} mean local lambda: {mean_lambda_loc[member]}")
                #     ax.plot(time, mean_lambda_loc[member], color = "blue", lw=2.0, label=f"van Loon+2026 {label}_{member}")
                #     ax.legend(fontsize=7, frameon=False)
                replot_mean = False
            else:      
                print("Flattening mean")
                mean_lambda_loc = mean_lambda_loc.flatten()
        if replot_mean:
            if label =="reconstructions":
                lw = 1.0
            else:
                lw = 0.3
            #mean_lambda_loc = rolling_mean_timeseries(pd.Series(mean_lambda_loc), window=ROLLING_WINDOW).values
            ax.plot(time, mean_lambda_loc, color = "blue", lw=lw, alpha = 0.7, label=f"van Loon+2026 {label}")
            ax.legend(fontsize=7, frameon=False)

        #sys.exit(4)

def read_in_and_overlay_temperature_timeseries(ax_temp):
    temperature_data = pd.read_csv("../data/calibration_data_Sep2025/annual_averages.csv")
    ax_temp.plot(temperature_data["timebound_lower"], temperature_data["GMST"], color="black", lw=2.0, label="Observed (annual avg)")


def make_scen7_temp_lambda_figures(
    internal_lambda=True,
    *,
    ladder=None,
    include_pdo: bool = True,
    rung_display_labels: dict[str, str] | None = None,
    rung_colours: dict[str, str] | None = None,
    output_suffix: str = "",
) -> None:
    """One two-panel figure per scen7 scenario.

    Left panel: 5-95% plume (median overlaid) of the blended surface
    temperature change. Right panel: 5-95% plume of the local feedback
    (30-yr centred OLS slope of Heat Uptake on temperature). Ladder rungs are
    overlaid in the comparison-plot style; both panels run only to 2100.
    """
    for cat in SCEN7_CATEGORIES:
        fig, (ax_lam, ax_temp) = plt.subplots(1, 2, figsize=(12, 4.5))
        if not draw_scen7_temp_lambda_row(
            ax_temp,
            ax_lam,
            cat,
            internal_lambda=internal_lambda,
            ladder=ladder,
            include_pdo=include_pdo,
            rung_display_labels=rung_display_labels,
            rung_colours=rung_colours,
        ):
            plt.close(fig)
            continue

        fig.tight_layout()
        fig.savefig(
            figure_path(
                f"ladder_scen7_{cat}_temp_lambda.png",
                include_pdo=include_pdo,
                output_suffix=output_suffix,
            ),
            dpi=150,
        )
        plt.close(fig)


def make_ladder_overview_composite(
    data,
    rungs,
    rung_display_labels,
    rung_colours,
    targets,
    *,
    ladder=None,
    include_pdo: bool = True,
    output_suffix: str = "",
    overview_scen7_category: str = "VL",
) -> None:
    """Create the requested 3x2 composite ladder overview figure."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))

    draw_sensitivity_marginals(
        axes[0],
        data,
        rungs,
        rung_display_labels,
        rung_colours,
        targets,
        panel_labels=("a)", "b)"),
    )
    draw_projection_violins(
        axes[1],
        data,
        rungs,
        rung_display_labels,
        rung_colours,
        panel_labels=("c)", "d)"),
    )
    if not draw_scen7_temp_lambda_row(
        axes[2, 1],
        axes[2, 0],
        overview_scen7_category,
        ladder=ladder,
        include_pdo=include_pdo,
        rung_display_labels=rung_display_labels,
        rung_colours=rung_colours,
        panel_labels=("f)", "e)"),
    ):
        plt.close(fig)
        return

    fig.tight_layout()
    fig.savefig(
        figure_path(
            "ladder_overview_composite.png",
            include_pdo=include_pdo,
            output_suffix=output_suffix,
        ),
        dpi=150,
    )
    plt.close(fig)


def main(
    *,
    ladder=None,
    include_pdo: bool = True,
    label_overrides: dict[str, str] | None = None,
    colour_overrides: dict[str, str] | None = None,
    output_suffix: str = "",
    overview_scen7_category: str = "VL",
) -> None:
    ladder = active_ladder(ladder=ladder, include_pdo=include_pdo)
    if not include_pdo:
        print("excluding PDO ladder rungs from plots")

    data = {label: per_member(dump) for label, dump, _, _ in ladder}
    rungs = list(data.keys())
    rung_display_labels = {
        label: display_label(
            label,
            include_pdo=include_pdo,
            label_overrides=label_overrides,
        )
        for label in rungs
    }
    rung_colours = build_rung_colours(rungs, colour_overrides=colour_overrides)

    targets = {
        "ECS": [3, 2,5],
        "TCRE": [1.65, 1, 2.3],
        "TCR": [1.8, 1.2, 2.4],
    }
    # ── Figure 1: ECS and TCR marginals ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_sensitivity_marginals(
        axes,
        data,
        rungs,
        rung_display_labels,
        rung_colours,
        targets,
    )
    fig.suptitle("Climate sensitivity along the structural ladder")
    fig.tight_layout()
    fig.savefig(
        figure_path(
            "ladder_sensitivity_marginals.png",
            include_pdo=include_pdo,
            output_suffix=output_suffix,
        ),
        dpi=150,
    )
    plt.close(fig)

    # ── Figure 2: end-of-century projection distributions ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_projection_violins(axes, data, rungs, rung_display_labels, rung_colours)
    fig.suptitle("End-of-century warming along the structural ladder "
                 "(box: IQR; whiskers: 5-95%)")
    fig.tight_layout()
    fig.savefig(
        figure_path(
            "ladder_projection_violins.png",
            include_pdo=include_pdo,
            output_suffix=output_suffix,
        ),
        dpi=150,
    )
    plt.close(fig)

    # ── Figure 3: flat10 carbon-cycle metrics (TCRE, ZEC50) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, xlabel in (
        ("tcre", axes[0], r"TCRE (K EgC$^{-1}$)"),
        ("zec50", axes[1], r"ZEC$_{50}$ (K)"),
    ):
        for label, folder, suffix, _ in ladder:
            fm = elm.flat10_metrics(suffix)
            if fm is None or col not in fm:
                make_flat10_key_metrics(folder, suffix)
                fm = elm.flat10_metrics(suffix)
            ax.hist(fm[col].dropna(), bins=40, histtype="step", density=True,
                    lw=2, color=rung_colours[label], label=rung_display_labels[label])
            ax.axvline(fm[col].median(), color=rung_colours[label], ls=":", lw=1)
        if col.upper() in targets:
            t = targets[col.upper()]
            ax.axvline(t[0], color="k", ls="--", lw=1, label="IPCC median target")
            ax.axvline(t[1], color="k", ls=":", lw=1, label="IPCC 5-95% range")
            ax.axvline(t[2], color="k", ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(r"\texttt{flat10MIP} carbon-cycle metrics along the ladder"
                 .replace(r"\texttt{", "").replace("}", ""))
    fig.tight_layout()
    fig.savefig(
        figure_path(
            "ladder_flat10_metrics.png",
            include_pdo=include_pdo,
            output_suffix=output_suffix,
        ),
        dpi=150,
    )
    plt.close(fig)

    # make_seferian3_ladder_figure(include_pdo=include_pdo)
    # make_gregory_ecs_plume_figure(include_pdo=include_pdo)
    make_scen7_temp_lambda_figures(
        include_pdo=include_pdo,
        ladder=ladder,
        rung_display_labels=rung_display_labels,
        rung_colours=rung_colours,
        output_suffix=output_suffix,
    )
    make_ladder_overview_composite(
        data,
        rungs,
        rung_display_labels,
        rung_colours,
        targets,
        ladder=ladder,
        include_pdo=include_pdo,
        output_suffix=output_suffix,
        overview_scen7_category=overview_scen7_category,
    )

    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main(include_pdo="--exclude-pdo" not in sys.argv)

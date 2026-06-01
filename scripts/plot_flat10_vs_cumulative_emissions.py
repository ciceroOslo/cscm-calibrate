"""Plot flat10 output variables against cumulative CO2 emissions.

For each flat10-class scenario, load the input emissions file
``<scenario>_em_gases_vupdate_2024_WMO_added_new.txt`` from
``GRAFITE/temp_indata``, derive cumulative CO2 emissions (sum of all CO2
columns, cumulatively summed over time), and plot it against ensemble
output variables from
``GRAFITE/cscm-calibrate/scripts/out_file_dump_new_preCO2/<scenario>_rcmip_draw_samples_500.csv``.

Outputs go to ``figures/flat10/cumulative_emissions/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_various_from_experiments import (
    DUMP_NEW,
    DUMP_OLD,
    DUMP_WIDE,
    FIG_FLAT10,
    FLAT10_SCENARIOS,
    extract_ensemble,
    load_scenario_csv,
    percentile_band,
    slugify,
)

INDATA_DIR = Path("/div/no-backup-nac/users/masan/GRAFITE/temp_indata")
INPUT_SUFFIX = "_em_gases_vupdate_2024_WMO_added_new.txt"
FIG_OUT = FIG_FLAT10 / "cumulative_emissions"

SOURCES = {"nopattern_noefficacy": DUMP_NEW, "nopattern": DUMP_OLD, "pattern": DUMP_WIDE}
SOURCE_COLOURS = {"nopattern_noefficacy": "C0", "nopattern": "C1", "pattern": "C2"}

# Map output-CSV scenario name -> emissions input file stem in INDATA_DIR.
# Defaults to identity (so flat10 scenarios just use their own name).
EMISSIONS_STEM_OVERRIDES = {
    "esm-hist": "historical",
}

SCENARIOS = list(FLAT10_SCENARIOS) + ["esm-hist"]

# Variables from the output CSV to plot against cumulative emissions
VARIABLES_TO_PLOT = [
    "Surface Air Temperature Change",
    "Surface Air Ocean Blended Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic|CO2",
    "Heat Content|Ocean",
    "Heat Uptake",
    "Carbon Pool|Land",
    "Carbon Pool|Ocean",
    "Net Flux to Atmosphere|CO2",
]


def load_cumulative_co2(scenario: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (years, cumulative_CO2_PgC) for a scenario, or None if missing.

    Sums all columns labeled ``CO2`` in the Component row, then cumulatively
    sums the annual values over time. The file has 4 header rows
    (Component, Unit, Description, Reference) followed by ``year value...``.
    """
    stem = EMISSIONS_STEM_OVERRIDES.get(scenario, scenario)
    path = INDATA_DIR / f"{stem}{INPUT_SUFFIX}"
    if not path.exists():
        print(f"[cum_emis] missing: {path}")
        return None

    with path.open() as fh:
        comp_row = fh.readline().split()  # 'Component', 'CO2', 'CO2', ...

    co2_idx = [i for i, name in enumerate(comp_row) if name == "CO2"]
    if not co2_idx:
        print(f"[cum_emis] no CO2 columns in {path}")
        return None

    # Parse data rows (skip 4 header rows). Whitespace-delimited.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=4,
        header=None,
        engine="python",
    )
    years = df.iloc[:, 0].to_numpy(dtype=int)
    # Component row index 0 is the literal "Component"; data column k corresponds
    # to header column k. So CO2 indices found above are direct.
    co2_annual = df.iloc[:, co2_idx].to_numpy(dtype=float).sum(axis=1)
    co2_cum = np.cumsum(co2_annual)
    return years, co2_cum


def align_years(emis_years: np.ndarray, emis_values: np.ndarray, out_years: np.ndarray) -> np.ndarray:
    """Return cumulative emissions aligned to ``out_years``; missing years get NaN."""
    lookup = dict(zip(emis_years.tolist(), emis_values.tolist()))
    return np.array([lookup.get(int(y), np.nan) for y in out_years])


def plot_scenario(scenario: str) -> None:
    cum = load_cumulative_co2(scenario)
    if cum is None:
        return
    emis_years, emis_cum = cum

    dfs = {}
    for label, folder in SOURCES.items():
        df = load_scenario_csv(folder, scenario, label)
        if df is None:
            print(f"[cum_emis] no output CSV for {scenario} in {label}")
            continue
        dfs[label] = df
    if not dfs:
        return

    out_dir = FIG_OUT / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    available_vars = set().union(*(set(df["variable"].unique()) for df in dfs.values()))
    for variable in VARIABLES_TO_PLOT:
        if variable not in available_vars:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        unit = ""
        for label, df in dfs.items():
            result = extract_ensemble(df, variable)
            if result is None:
                continue
            years, arr = result
            x = align_years(emis_years, emis_cum, years)
            mask = ~np.isnan(x)
            if not mask.any():
                continue
            x_m = x[mask]
            arr_m = arr[:, mask]
            p05, p50, p95 = percentile_band(arr_m)
            colour = SOURCE_COLOURS.get(label, "C0")
            ax.fill_between(x_m, p05, p95, color=colour, alpha=0.25, linewidth=0)
            ax.plot(x_m, p50, color=colour, lw=1.5, label=f"{label} (median, 5–95%)")
            unit_series = df.loc[df["variable"] == variable, "unit"]
            if not unit_series.empty:
                unit = unit_series.iloc[0]
            plotted = True
        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Cumulative CO2 emissions (Pg C)")
        ax.set_ylabel(f"{variable} ({unit})" if unit else variable)
        ax.set_title(f"{scenario}: {variable} vs cumulative CO2")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{scenario}_{slugify(variable)}_vs_cumCO2.png", dpi=120)
        plt.close(fig)
    print(f"[cum_emis] {scenario}: wrote plots to {out_dir}")


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        plot_scenario(scenario)


if __name__ == "__main__":
    main()

import os
import glob

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP_DIR = os.path.join(os.path.dirname(__file__), "out_file_dump")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "sanity_plots")
RUN_ID = "_252_6152"


def load_csv(filepath):
    """Load a result CSV and return rows for the chosen run_id."""
    df = pd.read_csv(filepath, index_col=0)
    df = df[df["run_id"] == RUN_ID]
    return df


def scenario_label(filepath):
    """Extract a human-readable scenario label from the filename."""
    base = os.path.basename(filepath)
    # strip the _rcmip_<json_stem>.csv suffix
    label = base.split("_rcmip_")[0]
    return label


def year_columns(df):
    """Return the year column names (integers) and their string keys."""
    meta_cols = ["climate_model", "model", "run_id", "scenario", "region", "variable", "unit"]
    year_cols = [c for c in df.columns if c not in meta_cols]
    return year_cols


def plot_per_variable(files):
    """
    For each unique variable found across the esm-allGHG-ssp files,
    produce one plot with one line per scenario.
    """
    # First pass: collect data keyed by variable
    var_data = {}  # variable -> list of (label, years, values, unit)
    for fpath in sorted(files):
        df = load_csv(fpath)
        if df.empty:
            continue
        label = scenario_label(fpath)
        ycols = year_columns(df)
        for _, row in df.iterrows():
            varname = row["variable"]
            unit = row["unit"]
            values = row[ycols].values.astype(float)
            years = np.array([int(y) for y in ycols])
            if varname not in var_data:
                var_data[varname] = []
            var_data[varname].append((label, years, values, unit))

    print(f"Found {len(var_data)} unique variables across esm-allGHG-ssp files")

    var_plot_dir = os.path.join(PLOT_DIR, "per_variable")
    os.makedirs(var_plot_dir, exist_ok=True)

    for varname, entries in sorted(var_data.items()):
        fig, ax = plt.subplots(figsize=(12, 5))
        for label, years, values, unit in entries:
            ax.plot(years, values, label=label, linewidth=0.8)
        ax.set_title(varname, fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel(entries[0][3])  # unit from first entry
        ax.legend(fontsize=6, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        # sanitise variable name for filename
        safe_name = varname.replace("|", "_").replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(var_plot_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved per-variable plots to {var_plot_dir}")
    


def plot_four_panel(files):
    """
    For each output file, produce a separate 2x2 panel plot showing:
      1) Effective Radiative Forcing|Anthropogenic|CO2
      2) Effective Radiative Forcing|Anthropogenic  +  Effective Radiative Forcing
      3) Surface Air Temperature Change
      4) Heat Content|Ocean
    """
    panel_vars = [
        (0, "Effective Radiative Forcing|Anthropogenic|CO2", "-"),
        (1, "Effective Radiative Forcing|Anthropogenic", "-"),
        (1, "Effective Radiative Forcing", "--"),
        (2, "Surface Air Temperature Change", "-"),
        (3, "Heat Content|Ocean", "-"),
    ]
    panel_titles = [
        "Effective Radiative Forcing|Anthropogenic|CO2",
        "ERF|Anthropogenic (solid) & ERF (dashed)",
        "Surface Air Temperature Change",
        "Heat Content|Ocean",
    ]
    panel_units = ["W/m²", "W/m²", "K", "ZJ"]

    four_panel_dir = os.path.join(PLOT_DIR, "four_panel")
    os.makedirs(four_panel_dir, exist_ok=True)

    for fpath in sorted(files):
        df = load_csv(fpath)
        if df.empty:
            continue
        label = scenario_label(fpath)
        ycols = year_columns(df)
        years = np.array([int(y) for y in ycols])

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(label, fontsize=12)
        axes = axes.flatten()

        for panel_idx, varname, ls in panel_vars:
            match = df[df["variable"] == varname]
            if match.empty:
                continue
            values = match.iloc[0][ycols].values.astype(float)
            axes[panel_idx].plot(years, values, ls, linewidth=1.0, label=varname)

        for i, ax in enumerate(axes):
            ax.set_title(panel_titles[i], fontsize=9)
            ax.set_xlabel("Year")
            ax.set_ylabel(panel_units[i])
            ax.grid(True, alpha=0.3)
            if i == 1:
                ax.legend(fontsize=6, loc="best")

        fig.tight_layout()
        safe_label = label.replace("|", "_").replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(four_panel_dir, f"{safe_label}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {len(files)} 4-panel plots to {four_panel_dir}")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    esm_allghg_ssp_files = sorted(glob.glob(os.path.join(DUMP_DIR, "esm-allGHG-ssp*_rcmip_*.csv")))
    all_files = sorted(glob.glob(os.path.join(DUMP_DIR, "*_rcmip_*.csv")))

    print(f"esm-allGHG-ssp files: {len(esm_allghg_ssp_files)}")
    print(f"All files: {len(all_files)}")

    print("\n--- Per-variable plots (esm-allGHG-ssp) ---")
    plot_per_variable(esm_allghg_ssp_files)

    print("\n--- 4-panel overview (all files) ---")
    plot_four_panel(all_files)

    print("\nDone.")

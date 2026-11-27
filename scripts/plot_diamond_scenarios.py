from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

from plot_various_from_experiments import extract_ensemble, percentile_band, plot_band, slugify


SCRIPT_DIR = Path(__file__).resolve().parent
DUMP_DIR = SCRIPT_DIR / "out_file_dump_nopattern_noefficacy"
FIG_ROOT = SCRIPT_DIR / "figures"
FIG_DIAMOND = FIG_ROOT / "diamond"
YSTART = 2000
MAX_YEND = 2100

CONFIG_NAME = "draw_samples_no_efficacy_no_pattern_wide_lambda_400"
CSV_SUFFIX = f"_infilled_DIAMOND_{CONFIG_NAME}.csv"
VARIABLE = "Surface Air Temperature Change"

MODEL_PATTERNS = {
    "gemini": "GEMINI-E3-EU",
    "nemesis": "NEMESIS-World",
    "omnia": "OMNIA",
    "gcam": "GCAM-Europe",
    "open-prom": "OPEN-PROM",
    "d5.5lab": "GEMINI-E3-EU"
}

model_colours = {
    "GEMINI-E3-EU": "C3",
    "NEMESIS-World": "C0",
    "OMNIA": "C1",
    "GCAM-Europe": "C2",
    "OPEN-PROM": "C4"
}

baseline_scenarionames = [
    'baseline_v1_2_new+softlink_omnia', 
    'diamond_baseline_gcameurope8.716102026', 
    'diamond_baseline_open-prom2.2', 
    'scenarioprotocolfulltrading_gemini-e3eu1.0', 
    'new_diam_cp_v20260904_nemesis-worldv1.1'
    ]

scenarios_no_pol = [
    'esm-allGHG-new_diam_nopol_v20260904_nemesis-worldv1.1',

]
# Scenario name (without the "esm-allGHG-" prefix) -> plot label
diamond_scen_name_mappings = {
    # NEMESIS-World v1.1
    "new_diam_cp_v20260904_nemesis-worldv1.1": "NDC_LTT",
    "new_diam_nopol_v20260904_nemesis-worldv1.1": "No_Pol",
    # OMNIA
    "baseline_v1_2_new+softlink_omnia": "NDC_LTT",
    "nopolicy_omnia": "No_Pol",
    "ltt_highce_omnia_eu": "LTT_HiCE (EU)",
    "ltt_highce_omnia_gbl": "LTT_HiCE (Global)",
    "ltt_medce_omnia_eu": "LTT_MedCE (EU)",
    "ltt_medce_omnia_gbl": "LTT_MedCE (Global)",
    "ltt_noce_omnia": "LTT_NoCE",
    "ndc_highce_omnia_eu": "NDC_HiCE (EU)",
    "ndc_highce_omnia_gbl": "NDC_HiCE (Global)",
    "ndc_medce_omnia_eu": "NDC_MedCE (EU)",
    "ndc_medce_omnia_gbl": "NDC_MedCE (Global)",
    "ndc_noce_omnia": "NDC_NoCE",
    # GCAM Europe 8.716102026
    "climpol_gcameurope8.716102026": "ClimPol",
    "diamond_baseline_gcameurope8.716102026": "NDC_LTT",
    "ref_gcameurope8.716102026": "No_Pol",
    # GEMINI-E3 EU 1.0
    "scenarionopolicy_gemini-e3eu1.0": "No_Pol",
    "scenarioprotocolesrtaxes+ets_gemini-e3eu1.0": "ESR_Tax+ETS",  # no label in table
    "scenarioprotocolets+ets-2_gemini-e3eu1.0": "ETS+ETS-2",  # no label in table
    "scenarioprotocolfulltrading_gemini-e3eu1.0": "NDC_LTT",
    "d5.5labnetzerocentral": "Lab_Cen",
    "d5.5labnetzeroflexible": "Lab_Flex",
    "d5.5labnetzerorigid": "Lab_Rgd",
    "d5.5labreference": "Lab_Ref",  # not in table
    # OPEN-PROM 2.2
    "diamond_baseline_open-prom2.2": "NDC_LTT",
    "npi_open-prom2.2": "ClimPol",
}


def percentile_band_custom(arr: np.ndarray, percentiles: np.ndarray) -> tuple:
    """Compute (p05, p50, p95) along axis 0."""
    percentiles_out = np.nanpercentile(arr, percentiles, axis=0)
    return percentiles_out

def list_diamond_scenarios(folder: Path) -> list[str]:
    scenarios = []
    for path in sorted(folder.glob(f"*{CSV_SUFFIX}")):
        stem = path.name.removesuffix(CSV_SUFFIX)
        if any(pattern in stem.lower() for pattern in MODEL_PATTERNS.keys()):
            scenarios.append(stem)
    return scenarios


def load_scenario_csv(folder: Path, scenario: str) -> pd.DataFrame | None:
    path = folder / f"{scenario}{CSV_SUFFIX}"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def identify_model(scenario: str) -> str | None:
    lowered = scenario.lower()
    for model, pattern in MODEL_PATTERNS.items():
        if model in lowered:
            return pattern
    return None


def split_model_groups(scenarios: list[str]) -> dict[str, list[str]]:
    grouped = {model: [] for model in MODEL_PATTERNS.values()}
    for scenario in scenarios:
        model = identify_model(scenario)
        if model is None:
            continue
        grouped[model].append(scenario)
    return {model: items for model, items in grouped.items() if items}


def longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for text in strings[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def longest_common_suffix(strings: list[str]) -> str:
    reversed_strings = [text[::-1] for text in strings]
    return longest_common_prefix(reversed_strings)[::-1]


def clean_label_fragment(text: str) -> str:
    if text in baseline_scenarionames:
        return "NDC LTT"
    cleaned = text.strip("_+- ")
    cleaned = re.sub(r"[_+]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip() or "baseline"


def make_scenario_labels(scenarios: list[str]) -> dict[str, str]:
    if len(scenarios) == 1:
        scenario = scenarios[0]
        label = clean_label_fragment(scenario.replace("esm-allGHG-", ""))
        if label in diamond_scen_name_mappings:
            label = diamond_scen_name_mappings[label]
        print(scenario, label)
        return {scenario: label}

    comparable = [scenario.replace("esm-allGHG-", "") for scenario in scenarios]
    common_prefix = longest_common_prefix(comparable)
    common_suffix = longest_common_suffix(comparable)

    labels = {}
    print("Common prefix:", common_prefix)
    print(scenarios)
    for scenario, comparable_name in zip(scenarios, comparable):
        if comparable_name in diamond_scen_name_mappings:
            labels[scenario] = diamond_scen_name_mappings[comparable_name]
            continue
        start = len(common_prefix)
        end = len(comparable_name) - len(common_suffix) if common_suffix else len(comparable_name)
        fragment = comparable_name[start:end]
        label = clean_label_fragment(fragment)
        if label in diamond_scen_name_mappings:
            label = diamond_scen_name_mappings[label]
            labels[scenario] = label
            continue
        print(comparable_name, fragment, label)
        print(scenario)
        sys.exit(4)
        if label == "baseline":
            label = clean_label_fragment(comparable_name)
        if scenario.replace("esm-allGHG-", "") in baseline_scenarionames:
            label = "NDC LTT"
        labels[scenario] = label
    return labels


def write_median_warming_member(
    folder: Path,
    scenario: str,
    years: np.ndarray,
    arr: np.ndarray,
) -> Path:
    final_year_values = arr[:, -1]
    valid_members = np.flatnonzero(~np.isnan(final_year_values))
    if valid_members.size == 0:
        raise ValueError(f"No valid end-of-run warming values found for {scenario}")

    ranked_members = valid_members[np.argsort(final_year_values[valid_members], kind="stable")]
    median_member_index = ranked_members[(ranked_members.size - 1) // 2]

    out_path = folder / f"{scenario}{CSV_SUFFIX.removesuffix('.csv')}_median_warming_w_infilling.csv"
    pd.DataFrame(
        {
            "years": years,
            "warming": arr[median_member_index],
        }
    ).to_csv(out_path, index=False)
    return out_path


def make_model_plot(
        model: str, 
        scenarios: list[str], 
        folder: Path, 
        out_dir: Path,
        just_ndc: bool = False
    ) -> Path | None:
    labels = make_scenario_labels(scenarios)
    print(labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    unit = ""

    for colour_index, scenario in enumerate(sorted(scenarios)):
        if just_ndc and labels[scenario] != "NDC LTT":
            continue
        df = load_scenario_csv(folder, scenario)
        if df is None:
            print(f"[diamond] missing CSV for {scenario}")
            continue
        data = extract_ensemble(df, VARIABLE)
        if data is None:
            print(f"[diamond] missing variable '{VARIABLE}' for {scenario}")
            continue
        years, arr = data
        median_member_path = write_median_warming_member(folder, scenario, years, arr)
        print(f"[diamond] wrote {median_member_path}")
        p05, p50, p95 = percentile_band(arr)
        print(labels)
        plot_band(ax, years, p05, p50, p95, color=f"C{colour_index}", label=labels[scenario])
        unit_series = df.loc[df["variable"] == VARIABLE, "unit"]
        if not unit and not unit_series.empty:
            unit = unit_series.iloc[0]
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(f"{model} scenarios — {VARIABLE}")
    ax.set_xlabel("Year")
    ax.set_xlim(left=YSTART, right=np.min([MAX_YEND, years[-1]]))
    ax.set_ylabel(f"{VARIABLE} ({unit})" if unit else VARIABLE)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    if just_ndc:
        out_path = out_dir / f"{slugify(model)}_{slugify(VARIABLE)}_w_infilling_ndc.png"
    else:
        out_path = out_dir / f"{slugify(model)}_{slugify(VARIABLE)}_w_infilling.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

def make_quantile_csv(model: str, scenarios: list[str], folder: Path, out_dir: Path) -> Path | None:
    quantiles = [5, 25, 50, 75, 95]
    all_data = []

    for scenario in sorted(scenarios):
        df = load_scenario_csv(folder, scenario)
        if df is None:
            print(f"[diamond] missing CSV for {scenario}")
            continue
        data = extract_ensemble(df, VARIABLE)
        if data is None:
            print(f"[diamond] missing variable '{VARIABLE}' for {scenario}")
            continue
        years, arr = data
        percentiles_out = percentile_band_custom(arr, quantiles)

        variable_rows = df.loc[df["variable"] == VARIABLE]
        if variable_rows.empty:
            print(f"[diamond] missing template row for '{VARIABLE}' in {scenario}")
            continue

        out_columns = [column for column in df.columns if column != "run_id"]
        template_row = variable_rows.iloc[0].drop(labels="run_id").copy()

        scenario_rows = []
        for quantile, values in zip(quantiles, percentiles_out):
            row = template_row.copy()
            row["variable"] = f"{VARIABLE}|{quantile}th quantile"
            row.loc[[str(year) for year in (years)]] = values
            scenario_rows.append(row)

        all_data.append(pd.DataFrame(scenario_rows, columns=out_columns))
            

    if not all_data:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(model)}_{slugify(VARIABLE)}_w_infilling_quantiles.csv"
    pd.concat(all_data, ignore_index=True).to_csv(out_path, index=False)
    return out_path


def make_year_bar_plot(
    grouped: dict[str, list[str]],
    folder: Path,
    out_dir: Path,
    year: int = 2050,
    just_ndc: bool = False,
) -> Path | None:
    """Floating range chart of warming in ``year`` for every scenario: a translucent
    box spans the 5th–95th percentile and a saturated horizontal line marks the
    median. Scenarios from the same model are placed together and share a colour."""
    #model_colours = {model: f"C{index}" for index, model in enumerate(MODEL_PATTERNS)}
    entries = []
    unit = ""

    for model, scenarios in grouped.items():
        labels = make_scenario_labels(scenarios)
        for scenario in sorted(scenarios):
            if just_ndc and labels[scenario] != "NDC LTT":
                continue
            df = load_scenario_csv(folder, scenario)
            if df is None:
                print(f"[diamond] missing CSV for {scenario}")
                continue
            data = extract_ensemble(df, VARIABLE)
            if data is None:
                print(f"[diamond] missing variable '{VARIABLE}' for {scenario}")
                continue
            years, arr = data
            if year not in years:
                continue
            year_matches = np.flatnonzero(years == year)
            if year_matches.size == 0:
                print(f"[diamond] year {year} not found for {scenario}")
                continue
            p05, p50, p95 = np.nanpercentile(arr[:, year_matches[0]], [5, 50, 95])
            entries.append((model, labels[scenario], p05, p50, p95))
            unit_series = df.loc[df["variable"] == VARIABLE, "unit"]
            if not unit and not unit_series.empty:
                unit = unit_series.iloc[0]

    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(entries) + 2), 5))
    positions = np.arange(len(entries))
    bar_width = 0.6
    seen_models = set()
    for position, (model, label, p05, p50, p95) in zip(positions, entries):
        colour = model_colours[model]
        ax.bar(
            position,
            p95 - p05,
            bottom=p05,
            width=bar_width,
            color=colour,
            alpha=0.35,
            linewidth=0,
            label=model if model not in seen_models else None,
        )
        ax.hlines(
            p50,
            position - bar_width / 2,
            position + bar_width / 2,
            color=colour,
            lw=2.5,
        )
        seen_models.add(model)

    ax.set_xticks(positions)
    if just_ndc:
        ax.set_xticklabels([f"{model}\n ({p50:.2f})" for model, label, _, p50, _  in entries if label == "NDC LTT"])
        ax.set_title(f"{VARIABLE} in {year} NDC LTT— median with 5th–95th range")
    else:
        ax.set_xticklabels([label for _, label, *_ in entries], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{VARIABLE} in {year} — median with 5th–95th percentile range")
    ax.set_ylabel(f"{VARIABLE} ({unit})" if unit else VARIABLE)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    if just_ndc:
        out_path = out_dir / f"all_models_{slugify(VARIABLE)}_{year}_bars_w_infilling_ndcc.png"
    else:
        out_path = out_dir / f"all_models_{slugify(VARIABLE)}_{year}_bars_w_infilling.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

def make_timeseries_for_all_models(
    grouped: dict[str, list[str]],
    dump_dir: Path,
    output_dir: Path,
    just_ndc: bool = False,
) -> Path | None:
    """Overlay scenario medians and 5th–95th percentile plumes across models."""
    #model_colours = {model: f"C{index}" for index, model in enumerate(MODEL_PATTERNS)}
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    unit = ""
    last_year = YSTART
    print(grouped.items())
    sys.exit(4)

    for model, scenarios in grouped.items():
        labels = make_scenario_labels(scenarios)
        for scenario in sorted(scenarios):
            if just_ndc and labels[scenario] != "NDC LTT":
                continue
            df = load_scenario_csv(dump_dir, scenario)
            if df is None:
                print(f"[diamond] missing CSV for {scenario}")
                continue
            data = extract_ensemble(df, VARIABLE)
            if data is None:
                print(f"[diamond] missing variable '{VARIABLE}' for {scenario}")
                continue
            years, arr = data
            p05, p50, p95 = percentile_band(arr)
            plot_band(
                ax, years, p05, p50, p95,
                color=model_colours[model],
                label=f"{model}: {labels[scenario]}",
            )
            unit_series = df.loc[df["variable"] == VARIABLE, "unit"]
            if not unit and not unit_series.empty:
                unit = unit_series.iloc[0]
            last_year = max(last_year, years[-1])
            plotted += 1

    if plotted == 0:
        print("[diamond] no matching scenario timeseries to plot across models")
        plt.close(fig)
        return None

    scenario_title = "NDC LTT" if just_ndc else "All scenarios"
    ax.set_title(f"{scenario_title} across models — {VARIABLE} (5th–95th percentile)")
    ax.set_xlabel("Year")
    ax.set_xlim(left=YSTART, right=min(MAX_YEND, last_year))
    ax.set_ylabel(f"{VARIABLE} ({unit})" if unit else VARIABLE)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_ndc" if just_ndc else ""
    out_path = output_dir / f"all_models_{slugify(VARIABLE)}_timeseries_w_infilling{suffix}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=DUMP_DIR,
        help="Directory containing Diamond scenario distribution CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIG_DIAMOND,
        help="Directory where grouped model figures will be written.",
    )
    parser.add_argument(
        "--bar-year",
        type=int,
        default=2050,
        help="Year for the cross-model median / 5th–95th percentile bar chart.",
    )
    args = parser.parse_args()

    scenarios = list_diamond_scenarios(args.dump_dir)
    print(scenarios)

    if not scenarios:
        raise FileNotFoundError(
            f"No Diamond output CSVs matching '*{CSV_SUFFIX}' found in {args.dump_dir}"
        )

    grouped = split_model_groups(scenarios)
    just_ndc = False
    if not grouped:
        raise RuntimeError("No model-tagged Diamond scenarios were found in the output dump.")
    start_omnia = False    
    for model, model_scenarios in grouped.items():
        if model == "omnia":
            print(model_scenarios)
            drop_eu = []
            for scenario in model_scenarios:
                print(scenario)
                if "omnia_eu" in scenario:
                    print("Found omnia_eu scenario")
                    drop_eu.append(scenario)
            for scenario in drop_eu:
                model_scenarios.remove(scenario)
            start_omnia = True
        # elif start_omnia:
        #     sys.exit(4)
        out_path = make_model_plot(model, model_scenarios, args.dump_dir, args.output_dir, just_ndc = just_ndc)
        make_quantile_csv(model, model_scenarios, args.dump_dir, args.output_dir)
        if out_path is not None:
            print(f"[diamond] wrote {out_path}")
    if just_ndc:
        timeseries_path = make_timeseries_for_all_models(
            grouped, args.dump_dir, args.output_dir, just_ndc=True
        )
        if timeseries_path is not None:
            print(f"[diamond] wrote {timeseries_path}")

    bar_path = make_year_bar_plot(grouped, args.dump_dir, args.output_dir, year=args.bar_year, just_ndc=just_ndc)
    if bar_path is not None:
        print(f"[diamond] wrote {bar_path}")
    bar_path = make_year_bar_plot(grouped, args.dump_dir, args.output_dir, year=2100, just_ndc=just_ndc)
    if bar_path is not None:
        print(f"[diamond] wrote {bar_path}")

    #model_colours = {model: f"C{index}" for index, model in enumerate(MODEL_PATTERNS)}
    for model, colour in model_colours.items():
        print(model)
        print(to_rgb(colour))
    print(model_colours)



if __name__ == "__main__":
    main()
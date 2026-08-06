from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_various_from_experiments import extract_ensemble, percentile_band, plot_band, slugify


SCRIPT_DIR = Path(__file__).resolve().parent
DUMP_DIR = SCRIPT_DIR / "out_file_dump_nopattern_noefficacy"
FIG_ROOT = SCRIPT_DIR / "figures"
FIG_DIAMOND = FIG_ROOT / "diamond"

CONFIG_NAME = "draw_samples_no_efficacy_no_pattern_wide_lambda_400"
CSV_SUFFIX = f"_idealised_{CONFIG_NAME}.csv"
VARIABLE = "Surface Air Temperature Change"

MODEL_PATTERNS = {
    "nemesis": "nemesis",
    "omnia": "omnia",
    "gcam": "gcameurope",
    "gemini": "gemini",
    "open-prom": "open-prom",
}


def list_diamond_scenarios(folder: Path) -> list[str]:
    scenarios = []
    for path in sorted(folder.glob(f"*{CSV_SUFFIX}")):
        stem = path.name.removesuffix(CSV_SUFFIX)
        if any(pattern in stem.lower() for pattern in MODEL_PATTERNS.values()):
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
        if pattern in lowered:
            return model
    return None


def split_model_groups(scenarios: list[str]) -> dict[str, list[str]]:
    grouped = {model: [] for model in MODEL_PATTERNS}
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
    cleaned = text.strip("_+- ")
    cleaned = re.sub(r"[_+]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip() or "baseline"


def make_scenario_labels(scenarios: list[str]) -> dict[str, str]:
    if len(scenarios) == 1:
        scenario = scenarios[0]
        label = clean_label_fragment(scenario.replace("esm-allGHG-", ""))
        return {scenario: label}

    comparable = [scenario.replace("esm-allGHG-", "") for scenario in scenarios]
    common_prefix = longest_common_prefix(comparable)
    common_suffix = longest_common_suffix(comparable)

    labels = {}
    for scenario, comparable_name in zip(scenarios, comparable):
        start = len(common_prefix)
        end = len(comparable_name) - len(common_suffix) if common_suffix else len(comparable_name)
        fragment = comparable_name[start:end]
        label = clean_label_fragment(fragment)
        if label == "baseline":
            label = clean_label_fragment(comparable_name)
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

    out_path = folder / f"{scenario}{CSV_SUFFIX.removesuffix('.csv')}_median_warming.csv"
    pd.DataFrame(
        {
            "years": years,
            "warming": arr[median_member_index],
        }
    ).to_csv(out_path, index=False)
    return out_path


def make_model_plot(model: str, scenarios: list[str], folder: Path, out_dir: Path) -> Path | None:
    labels = make_scenario_labels(scenarios)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    unit = ""

    for colour_index, scenario in enumerate(sorted(scenarios)):
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
    ax.set_ylabel(f"{VARIABLE} ({unit})" if unit else VARIABLE)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(model)}_{slugify(VARIABLE)}.png"
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
    args = parser.parse_args()

    scenarios = list_diamond_scenarios(args.dump_dir)
    if not scenarios:
        raise FileNotFoundError(
            f"No Diamond output CSVs matching '*{CSV_SUFFIX}' found in {args.dump_dir}"
        )

    grouped = split_model_groups(scenarios)
    if not grouped:
        raise RuntimeError("No model-tagged Diamond scenarios were found in the output dump.")

    for model, model_scenarios in grouped.items():
        out_path = make_model_plot(model, model_scenarios, args.dump_dir, args.output_dir)
        if out_path is not None:
            print(f"[diamond] wrote {out_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Generate the ladder figures for a user-configured set of experiments.

Edit ``ACTIVE_LADDER`` to choose which ladder experiments to include. Entries
use the same 4-tuple structure as ``extract_ladder_metrics.LADDER``:
``(label, dump_directory_name, flat10_metrics_suffix, status)``.

Edit ``OUTPUT_SUFFIX`` to append a suffix to all generated plot filenames.
"""
from __future__ import annotations

import extract_ladder_metrics as elm
import plot_ladder_figures_v3 as ladder_plots


# Configure the experiments to include in this plot bundle.
# You can subset ``elm.LADDER`` or append new ladder-style experiment tuples.
ACTIVE_LADDER = list(elm.LADDER)
print(ACTIVE_LADDER)
new_exps = [
    ('5 +PDOeff', 'out_file_dump_just_pdo_effic', 'pdoeff', 'done'),
    ('4c IPCC+PDOeff', 'out_file_dump_just_pdo_effic_ar7', 'pdoeff_ar7', 'done'),
    ('6 +PDOlam', 'out_file_dump_just_pdo_lambda', 'pdolambda', 'done'),
    ('4d IPCC+PDOlam', 'out_file_dump_just_pdo_lambda_ar7', 'pdolambda_ar7', 'done'),
            ]
for exp in new_exps:
    ACTIVE_LADDER.append(exp)
# Appended to all output figure names. Examples: "", "paper_subset", "test_a".
OUTPUT_SUFFIX = "varying_all"

# Optional display-name overrides for legends and axis labels.
DISPLAY_LABEL_OVERRIDES = {
    "0 baseline (v1)": "baseline",
    "1 +efficacy": "efficacy",
    "2 +pattern": "aerosol",
    "4a IPCC+pattern": "aerosol ECS constrained",
    "3 +PDO": " full PDO",
    "4b IPCC+pat+PDO": "full PDO ECS constrained",
    "5 +PDOeff": "PDOeff",
    "4c IPCC+PDOeff": "PDOeff ECS constrained",
    "6 +PDOlam": "PDOlam",
    "4d IPCC+PDOlam": "PDOlam ECS constrained",
}

# Optional colour overrides keyed by ladder label.
COLOUR_OVERRIDES = {
    # "New experiment": "#8c564b",
}

# Composite figure bottom-row scen7 category.
OVERVIEW_SCEN7_CATEGORY = "VL"


def main() -> None:
    ladder_plots.main(
        ladder=ACTIVE_LADDER,
        label_overrides=DISPLAY_LABEL_OVERRIDES or None,
        colour_overrides=COLOUR_OVERRIDES or None,
        output_suffix=OUTPUT_SUFFIX,
        overview_scen7_category=OVERVIEW_SCEN7_CATEGORY,
    )


if __name__ == "__main__":
    main()

"""
Make prior/posterior parameter tables (csv and LaTeX) and histogram panels.

All arguments are optional, defaults are set in the DEFAULT_* constants below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cscm_calibrate.parameter_table import (  # noqa: E402
    make_parameter_table,
    plot_parameter_histograms,
    write_table_csv,
    write_table_latex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = REPO_ROOT / "tests" / "test-data" / "config_file.json"
DEFAULT_SAMPLES = [
    REPO_ROOT / "draw_samples_archive" / "draw_samples_no_efficacy_no_pattern_wide_lambda_400.json",
    REPO_ROOT / "draw_samples_archive" / "draw_samples_no_delta_aero_wide_lambda_400.json",
    REPO_ROOT / "draw_samples_archive" / "draw_samples_delta_aero_and_efficacy_wide_lambda_500.json", 
    REPO_ROOT / "draw_samples_archive" / "draw_samples_ar7_v1_500.json",
    ]
DEFAULT_LABELS = [
    "baseline",
    "efficacy",
    "aerosol",
    "aerosol ECS constrained",
]
DEFAULT_VARIABLES = None
DEFAULT_PERCENTILES = (5.0, 95.0)
DEFAULT_SIG_DIGITS = 3
DEFAULT_OUTDIR = REPO_ROOT / "parameter_table_output"
DEFAULT_FNAME_STUB = "parameter_table.csv"
DEFAULT_BINS = 30
DEFAULT_NCOLS = 4


def parse_args(argv=None):
    """Parse command line arguments, all of which have defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=Path, nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    parser.add_argument(
        "--percentiles", type=float, nargs=2, default=DEFAULT_PERCENTILES
    )
    parser.add_argument("--sig-digits", type=int, default=DEFAULT_SIG_DIGITS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fname-stub", default=DEFAULT_FNAME_STUB)
    parser.add_argument("--no-hist", action="store_true")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--ncols", type=int, default=DEFAULT_NCOLS)
    return parser.parse_args(argv)


def main(argv=None):
    """Write the parameter table and histograms."""
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    table = make_parameter_table(
        args.config,
        samples_paths=args.samples,
        labels=args.labels,
        variables=args.variables,
        percentiles=tuple(args.percentiles),
        sig_digits=args.sig_digits,
    )
    write_table_csv(table, args.outdir / f"{args.fname_stub}.csv")
    write_table_latex(table, args.outdir / f"{args.fname_stub}.tex")
    print(table.to_string(index=False))

    if not args.no_hist:
        plot_parameter_histograms(
            args.samples,
            labels=args.labels,
            variables=args.variables,
            config_path=args.config,
            ncols=args.ncols,
            bins=args.bins,
            outfile=args.outdir / f"{args.fname_stub}.png",
        )
    print(f"Output written to {args.outdir}")


if __name__ == "__main__":
    main()

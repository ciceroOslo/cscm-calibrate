"""
Tables and histograms of prior ranges and posterior percentile ranges.

Prior ranges are read from ``prior_configs.prior_distro_dict`` in a
calibration config file, posterior samples from ``draw_samples_*.json`` files.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAMSET_KEYS = ("pamset_udm", "pamset_emiconc", "pamset_carbon")

LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _select_variables(available: Sequence[str], variables: Sequence[str] | None):
    if variables is None:
        return list(available)
    missing = [var for var in variables if var not in available]
    if missing:
        raise KeyError(f"Variables not found: {missing}")
    return list(variables)


def _load_prior_dict(config_path, raw=False):
    with open(config_path, encoding="utf-8") as fh:
        if raw:
            # Keep numbers as the literal strings written in the json file
            config = json.load(fh, parse_float=str, parse_int=str)
        else:
            config = json.load(fh)
    return config["prior_configs"]["prior_distro_dict"]


def read_prior_ranges(config_path, variables=None):
    """
    Read prior ranges from a calibration config file.

    Parameters
    ----------
    config_path : str or Path
        Path to a config json with ``prior_configs.prior_distro_dict``.
    variables : list of str, optional
        Subset of variables to return. Default is all, in config order.

    Returns
    -------
    dict
        Variable name -> [lower, upper] as floats.
    """
    prior_dict = _load_prior_dict(config_path)
    selected = _select_variables(list(prior_dict), variables)
    return {var: [float(val) for val in prior_dict[var]] for var in selected}


def format_range(lower, upper):
    """Format a range as ``"x-y"``."""
    return f"{lower}-{upper}"


def read_posterior_samples(samples_path, variables=None):
    """
    Read a draw_samples json file into a DataFrame.

    Parameters
    ----------
    samples_path : str or Path
        Path to a json list of samples, each with pamset dictionaries.
    variables : list of str, optional
        Subset of variables to return. Default is all.

    Returns
    -------
    pandas.DataFrame
        One row per sample, one column per parameter.
    """
    with open(samples_path, encoding="utf-8") as fh:
        samples = json.load(fh)
    rows = []
    for sample in samples:
        row = {}
        for key in PAMSET_KEYS:
            row.update(sample.get(key, {}))
        rows.append(row)
    samples_df = pd.DataFrame(rows)
    return samples_df[_select_variables(list(samples_df.columns), variables)]


def posterior_percentile_ranges(
    samples_df, variables=None, percentiles=(5, 95), sig_digits=3
):
    """
    Compute formatted percentile ranges for each variable.

    Parameters
    ----------
    samples_df : pandas.DataFrame
        Posterior samples, one column per parameter.
    variables : list of str, optional
        Variables to compute ranges for. Default is all columns.
        Variables not in ``samples_df`` get an empty string.
    percentiles : tuple of float
        Lower and upper percentile.
    sig_digits : int
        Significant digits in the formatted numbers.

    Returns
    -------
    dict
        Variable name -> ``"x-y"`` string.
    """
    if variables is None:
        variables = list(samples_df.columns)
    ranges = {}
    for var in variables:
        if var not in samples_df.columns:
            ranges[var] = ""
            continue
        lower, upper = np.percentile(samples_df[var].to_numpy(), percentiles)
        ranges[var] = format_range(f"{lower:.{sig_digits}g}", f"{upper:.{sig_digits}g}")
    return ranges


def _default_labels(samples_paths, labels):
    if labels is None:
        return [Path(path).stem for path in samples_paths]
    if len(labels) != len(samples_paths):
        raise ValueError("labels must have the same length as samples_paths")
    return list(labels)


def make_parameter_table(
    config_path,
    samples_paths=None,
    labels=None,
    variables=None,
    percentiles=(5, 95),
    sig_digits=3,
):
    """
    Make a table of prior ranges and posterior percentile ranges.

    Parameters
    ----------
    config_path : str or Path
        Config json with ``prior_configs.prior_distro_dict``.
    samples_paths : list of str or Path, optional
        draw_samples json files, one posterior column is added per file.
    labels : list of str, optional
        Column labels for the sample files. Default is the file stems.
    variables : list of str, optional
        Subset of variables. Default is all variables in the config.
    percentiles : tuple of float
        Lower and upper percentile for the posterior ranges.
    sig_digits : int
        Significant digits in the posterior ranges.

    Returns
    -------
    pandas.DataFrame
        Columns ``Variable name``, ``Prior range`` and
        one posterior column per sample file.
    """
    prior_dict = _load_prior_dict(config_path, raw=True)
    selected = _select_variables(list(prior_dict), variables)
    table = pd.DataFrame(
        {
            "Variable name": selected,
            "Prior range": [format_range(*prior_dict[var]) for var in selected],
        }
    )
    samples_paths = list(samples_paths or [])
    for path, label in zip(samples_paths, _default_labels(samples_paths, labels)):
        ranges = posterior_percentile_ranges(
            read_posterior_samples(path),
            variables=selected,
            percentiles=percentiles,
            sig_digits=sig_digits,
        )
        column = f"Posterior {percentiles[0]:g}-{percentiles[1]:g}% ({label})"
        table[column] = [ranges[var] for var in selected]
    return table


def write_table_csv(table, path):
    """Write the parameter table to csv."""
    table.to_csv(path, index=False)


def escape_latex(text):
    """Escape LaTeX special characters in a string."""
    return "".join(LATEX_SPECIAL_CHARS.get(char, char) for char in str(text))


def table_to_latex(table, caption=None, label=None):
    """
    Convert the parameter table to a LaTeX table string.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to convert.
    caption : str, optional
        Table caption.
    label : str, optional
        LaTeX label for the table.

    Returns
    -------
    str
        LaTeX source for the table.
    """
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption is not None:
        lines.append(rf"\caption{{{escape_latex(caption)}}}")
    if label is not None:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{" + "l" * len(table.columns) + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(escape_latex(col) for col in table.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in table.iterrows():
        lines.append(" & ".join(escape_latex(val) for val in row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table_latex(table, path, caption=None, label=None):
    """Write the parameter table to a LaTeX file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(table_to_latex(table, caption=caption, label=label))


def plot_parameter_histograms(
    samples_paths,
    labels=None,
    variables=None,
    config_path=None,
    ncols=4,
    bins=30,
    outfile=None,
    max_rows_per_file = 3
):
    """
    Plot histogram panels of posterior samples for each variable.

    Parameters
    ----------
    samples_paths : list of str or Path
        draw_samples json files, overlaid in each panel.
    labels : list of str, optional
        Legend labels for the sample files. Default is the file stems.
    variables : list of str, optional
        Subset of variables. Default is all variables in the config if
        given, otherwise all variables in the first sample file.
    config_path : str or Path, optional
        Config json, if given the prior range is shaded in each panel.
    ncols : int
        Number of panel columns.
    bins : int
        Number of histogram bins.
    outfile : str or Path, optional
        If given, the figure is saved here.

    Returns
    -------
    matplotlib.figure.Figure
        The histogram figure.
    """
    samples_paths = list(samples_paths)
    labels = _default_labels(samples_paths, labels)
    samples = [read_posterior_samples(path) for path in samples_paths]
    priors = read_prior_ranges(config_path) if config_path is not None else {}
    if variables is None:
        variables = list(priors) if priors else list(samples[0].columns)

    nrows = int(np.ceil(len(variables) / ncols))
    file_num = int(np.ceil(nrows / max_rows_per_file))
    print(file_num)

    fgs_list = []
    axs_list = []
    for i in range(int(file_num)):
        start_row = i * max_rows_per_file
        end_row = start_row + max_rows_per_file
        fig, axs = plt.subplots(
            nrows=min(max_rows_per_file, nrows - start_row),
            ncols=ncols,
            figsize=(4 * ncols, 3 * min(max_rows_per_file, nrows - start_row)),
            squeeze=False,
        )
        fgs_list.append(fig)
        axs_list.append(axs)
    for axnum, var in enumerate(variables):
        print(f"nrows: {nrows}, max_rows_per_file: {max_rows_per_file}, ncols: {ncols}, axnum: {axnum}")
        fignum = axnum // (ncols*max_rows_per_file)
        axnum_this_fig = axnum - fignum * max_rows_per_file * ncols
        print(f"Fig. {fignum}, axnum_this_fig: {axnum_this_fig}, ax shape: {axs_list[fignum].shape}")
        print(f"Placing on row {axnum_this_fig % max_rows_per_file}, column {axnum_this_fig // max_rows_per_file}")
        ax = axs_list[fignum][axnum_this_fig % max_rows_per_file][axnum_this_fig // max_rows_per_file]
        if var in priors:
            ax.axvspan(*priors[var], color="grey", alpha=0.2, label="Prior range")
        for samples_df, label in zip(samples, labels):
            if var in samples_df.columns:
                ax.hist(samples_df[var], bins=bins, alpha=0.5, label=label)
        ax.set_title(var)
    for ax in axs_list[-1].flat[len(variables)-ncols*max_rows_per_file*(len(axs_list)-1):]:
        ax.set_visible(False)

    legend_entries = {}
    for axs in axs_list:
        for ax in axs.flat:
            if not ax.get_visible():
                continue
            for handle, legend_label in zip(*ax.get_legend_handles_labels()):
                legend_entries.setdefault(legend_label, handle)
    for i, fig in enumerate(fgs_list):
        fig.legend(
            legend_entries.values(),
            legend_entries.keys(),
            loc="upper center",
            ncols=max(len(legend_entries), 1),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if outfile is not None:
            if len(fgs_list) > 1:
                outfile_now = str(outfile).replace(".png", f"_{i}.png")
            else:
                outfile_now = outfile
            fig.savefig(outfile_now)
    return fgs_list if len(fgs_list) > 1 else fgs_list[0]

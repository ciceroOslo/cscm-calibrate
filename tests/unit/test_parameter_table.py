"""Tests for the prior/posterior parameter table module."""
import json
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cscm_calibrate import parameter_table as pt  # noqa: E402

CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "test-data", "config_file_pdo.json"
)


def _write_samples(path, values):
    samples = [
        {
            "pamset_udm": {"lambda": lam, "lm": 40},
            "pamset_emiconc": {"qo3": qo3},
            "pamset_carbon": {"beta_f": beta_f},
            "Index": f"_{i}",
        }
        for i, (lam, qo3, beta_f) in enumerate(values)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh)
    return path


@pytest.fixture
def samples_file(tmp_path):
    values = [(0.5 + 0.01 * i, 0.4 + 0.002 * i, 0.3) for i in range(101)]
    return _write_samples(tmp_path / "draw_samples_test.json", values)


def test_read_prior_ranges_all_and_subset():
    priors = pt.read_prior_ranges(CONFIG)
    assert list(priors)[:3] == ["rlamdo", "akapa", "cpi"]
    assert priors["delta_lambda_aero"] == [-0.7, 3.4]

    subset = pt.read_prior_ranges(CONFIG, variables=["qo3", "lambda"])
    assert list(subset) == ["qo3", "lambda"]

    with pytest.raises(KeyError, match="not_a_param"):
        pt.read_prior_ranges(CONFIG, variables=["lambda", "not_a_param"])


def test_read_posterior_samples_flattens(samples_file):
    samples_df = pt.read_posterior_samples(samples_file)
    assert set(samples_df.columns) == {"lambda", "lm", "qo3", "beta_f"}
    assert len(samples_df) == 101

    subset = pt.read_posterior_samples(samples_file, variables=["qo3"])
    assert list(subset.columns) == ["qo3"]


def test_posterior_percentile_ranges(samples_file):
    samples_df = pt.read_posterior_samples(samples_file)
    ranges = pt.posterior_percentile_ranges(
        samples_df, variables=["lambda", "missing"]
    )
    assert ranges["lambda"] == "0.55-1.45"
    assert ranges["missing"] == ""


def test_make_parameter_table(samples_file, tmp_path):
    other = _write_samples(tmp_path / "other.json", [(1.0, 0.5, 0.3)] * 10)
    table = pt.make_parameter_table(
        CONFIG,
        samples_paths=[samples_file, other],
        labels=["A", "B"],
        variables=["lambda", "qindso2", "beto"],
    )
    assert list(table.columns) == [
        "Variable name",
        "Description",
        "Prior range",
        "Posterior 5-95% (A)",
        "Posterior 5-95% (B)",
    ]
    # Prior numbers are kept exactly as written in the json
    assert list(table["Prior range"]) == ["0.4-2.16", "-0.010--0.00", "0-7"]
    assert list(table["Description"]) == ["", "", ""]
    assert table["Posterior 5-95% (B)"].iloc[0] == "1-1"
    assert table["Posterior 5-95% (A)"].iloc[1] == ""


def test_make_parameter_table_default_labels(samples_file):
    table = pt.make_parameter_table(CONFIG, samples_paths=[samples_file])
    assert "Posterior 5-95% (draw_samples_test)" in table.columns
    assert len(table) == len(pt.read_prior_ranges(CONFIG))


def test_write_csv_and_latex(samples_file, tmp_path):
    table = pt.make_parameter_table(
        CONFIG, samples_paths=[samples_file], variables=["rb_coef0"]
    )
    pt.write_table_csv(table, tmp_path / "table.csv")
    pt.write_table_latex(table, tmp_path / "table.tex", caption="Test", label="tab:x")

    assert (tmp_path / "table.csv").read_text().startswith(
        "Variable name,Description,Prior range"
    )
    latex = (tmp_path / "table.tex").read_text()
    assert r"rb\_coef0" in latex
    assert r"5-95\%" in latex
    assert r"\label{tab:x}" in latex
    assert r"\begin{tabular}{llll}" in latex


def test_plot_parameter_histograms(samples_file, tmp_path):
    outfile = tmp_path / "hist.png"
    fig = pt.plot_parameter_histograms(
        [samples_file, samples_file],
        labels=["A", "B"],
        variables=["lambda", "qo3", "beta_f"],
        config_path=CONFIG,
        ncols=2,
        outfile=outfile,
    )
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert outfile.exists()
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend_labels == ["Prior range", "A", "B"]
    assert np.isclose(len(visible[0].patches), 1 + 2 * 30)

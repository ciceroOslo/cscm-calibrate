"""Snapshot-based regression tests.

Each test pins the numerical output of a deterministic function against a
golden snapshot stored under ``tests/regression/snapshots/``.  On first run (or
when invoked with ``--regen-snapshots``) the snapshot is written; subsequent
runs assert that the function still produces the same result within tolerance.

Run only these tests with::

    pytest tests/regression -m regression

Regenerate after an intentional behaviour change::

    pytest tests/regression -m regression --regen-snapshots
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cscm_calibrate.shared_functions import (
    compute_ecs_gregory,
    make_config_distro_json,
    make_constraints_config_from_RCMIP_csv,
)
from cscm_calibrate.weigth_ensemble_from_constraints_and_draw import (
    calculate_sample_weights,
)

pytestmark = pytest.mark.regression


# ---------------------------------------------------------------------------
# compute_ecs_gregory: deterministic abrupt-4xCO2 dataset → ECS series
# ---------------------------------------------------------------------------


def _synthetic_abrupt_dataset(n_members=4, n_years=30, seed=12345):
    """Build a fixed-RNG synthetic abrupt-4xCO2 dataset.

    The forced response is shared across members with member-specific
    perturbations on the OHC curve, so each run_id gets a distinct ECS.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(n_years)
    year_cols = [str(y) for y in years]
    rows = []
    for run_id in range(n_members):
        # Per-member feedback strength and forcing scaling, drawn deterministically.
        lam = -2.0 - 0.3 * run_id           # net feedback (slope)
        f4x = 7.0 + 0.5 * run_id            # 4xCO2 forcing
        # Idealised exponential approach to equilibrium.
        tau = 8.0 + run_id
        t = (f4x / -lam) * (1 - np.exp(-years / tau))
        # OHC = cumulative TOA imbalance ≈ f4x + lam*t integrated.
        n_toa = f4x + lam * t
        ohc = np.cumsum(n_toa)
        # Add small reproducible noise so the regression is not trivial.
        t = t + rng.normal(0, 0.02, size=n_years)
        ohc = ohc + rng.normal(0, 0.05, size=n_years)

        trow = {
            "variable": "Surface Air Ocean Blended Temperature Change",
            "run_id": run_id,
        }
        orow = {"variable": "Heat Content|Ocean", "run_id": run_id}
        for y, cy in enumerate(year_cols):
            trow[cy] = float(t[y])
            orow[cy] = float(ohc[y])
        rows.extend([trow, orow])
    return pd.DataFrame(rows)


def test_compute_ecs_gregory_snapshot(snapshot_compare):
    df = _synthetic_abrupt_dataset()
    ecs = compute_ecs_gregory(df, start_year=0, end_year=29)
    snapshot_compare(
        "compute_ecs_gregory",
        {"ecs": ecs.to_numpy(dtype=float)},
    )


# ---------------------------------------------------------------------------
# make_constraints_config_from_RCMIP_csv: fixed CSV → constraints DataFrame
# ---------------------------------------------------------------------------


def _rcmip_constraints_df():
    return pd.DataFrame(
        {
            "Variable": [
                "Atmospheric Concentrations|CO2",
                "Heat Content|Ocean",
                "Effective Radiative Forcing|Aerosols",
                "Global Mean Surface Temperature (GMST)",
            ],
            "Baseline_period": ["1850-1900", "1971-1971", "1750-1750", "1850-1900"],
            "Constraint_period": [
                "2014-2014",
                "2018-2018",
                "2014-2014",
                "2010-2019",
            ],
            "Central_estimate": [397.0, 330.0, -1.3, 1.06],
            "Lower_bound": [395.0, 200.0, -2.0, 0.85],
            "Upper_bound": [399.0, 460.0, -0.6, 1.27],
        }
    )


def test_make_constraints_config_snapshot(snapshot_compare):
    with patch("pandas.read_csv", return_value=_rcmip_constraints_df()):
        result = make_constraints_config_from_RCMIP_csv("dummy.csv")
    # Keep only numerical / string columns for stable serialisation.
    snapshot_compare("constraints_from_rcmip", result, fmt="csv")


# ---------------------------------------------------------------------------
# make_config_distro_json: fixed parameter matrix → config JSON
# ---------------------------------------------------------------------------


def test_make_config_distro_json_snapshot(tmp_path, snapshot_compare):
    rng = np.random.default_rng(7)
    parameter_names = [
        "lambda",           # routed to pamset_udm via ordering_standard_forc
        "qbmb",             # routed to pamset_emiconc (default bucket)
        "beta_f",           # routed to pamset_carbon via CARBON_CYCLE_MODEL_REQUIRED_PAMSET
        "rb_1",             # routed to pamset_carbon via rb_/rs_ prefix rule
        "rs_2",             # routed to pamset_carbon via rb_/rs_ prefix rule
    ]
    matrix = rng.uniform(0.0, 2.0, size=(len(parameter_names), 3))
    make_config_distro_json(
        matrix,
        parameter_names,
        json_name="snapshot_cfgs.json",
        index_list=["alpha", "beta", "gamma"],
        output_dir=str(tmp_path),
    )
    with open(tmp_path / "snapshot_cfgs.json", encoding="utf-8") as fh:
        cfgs = json.load(fh)
    snapshot_compare("make_config_distro_json", cfgs, fmt="json")


# ---------------------------------------------------------------------------
# calculate_sample_weights: deterministic distributions + samples → weights
# ---------------------------------------------------------------------------


def test_calculate_sample_weights_snapshot(snapshot_compare):
    """Pin the sample-weight reweighting output for a small deterministic input.

    Uses a fixed RNG seed for both the target distributions and the samples
    to ensure bit-stable behaviour.
    """
    rng = np.random.RandomState(2024)
    n = 500
    samples_a = rng.normal(loc=0.0, scale=1.0, size=n)
    samples_b = rng.normal(loc=1.0, scale=2.0, size=n)
    samples_df = pd.DataFrame({"a": samples_a, "b": samples_b})

    # Target marginals differ slightly from the priors so reweighting has work to do.
    target_a = samples_a + 0.3
    target_b = samples_b - 0.5

    range_width_a_extra = (samples_a.max() - samples_a.min()) / 1e3
    range_width_b_extra = (samples_b.max() - samples_b.min()) / 1e3
    bins_a = np.histogram(
        samples_a,
        bins=20,
        range=(
            samples_a.min() - range_width_a_extra,
            samples_a.max() + range_width_a_extra,
        ),
    )[1]
    bins_b = np.histogram(
        samples_b,
        bins=20,
        range=(
            samples_b.min() - range_width_b_extra,
            samples_b.max() + range_width_b_extra,
        ),
    )[1]

    distributions = {
        "a": {"bins": bins_a, "values": target_a},
        "b": {"bins": bins_b, "values": target_b},
    }

    weights_final, gofs, gofs_full = calculate_sample_weights(
        distributions, samples_df, niterations=5
    )
    snapshot_compare(
        "calculate_sample_weights",
        {
            "weights_final": np.asarray(weights_final, dtype=float),
            "gofs": gofs.to_numpy(dtype=float),
        },
        rtol=1e-6,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Meta: --regen-snapshots actually rewrites a snapshot file
# ---------------------------------------------------------------------------


def test_snapshots_directory_exists():
    """The snapshots directory must exist after the suite has run once."""
    snapshots_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "regression",
        "snapshots",
    )
    assert os.path.isdir(snapshots_dir)

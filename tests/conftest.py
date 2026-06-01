"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import json
import os.path

import numpy as np
import pandas as pd
import pytest

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")
SNAPSHOTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "regression", "snapshots"
)


@pytest.fixture(scope="session")
def test_data_dir():
    return TEST_DATA_DIR


# ---------------------------------------------------------------------------
# Snapshot infrastructure for regression tests
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Register the --regen-snapshots CLI flag."""
    parser.addoption(
        "--regen-snapshots",
        action="store_true",
        default=False,
        help="Regenerate regression snapshot files instead of comparing.",
    )


@pytest.fixture
def regen_snapshots(request):
    return request.config.getoption("--regen-snapshots")


@pytest.fixture
def snapshot_compare(regen_snapshots):
    """Return a callable that writes (first run / --regen-snapshots) or compares
    (subsequent runs) a snapshot file under tests/regression/snapshots/.

    Usage::

        snapshot_compare("name", {"arr": np.array([...])})              # NPZ
        snapshot_compare("name", {"foo": 1, "bar": [1, 2]}, fmt="json") # JSON
        snapshot_compare("name", dataframe, fmt="csv")                   # CSV
    """
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    def _compare(name, payload, fmt="npz", rtol=1e-6, atol=1e-9):
        if fmt == "npz":
            path = os.path.join(SNAPSHOTS_DIR, f"{name}.npz")
            arrays = {k: np.asarray(v) for k, v in payload.items()}
            if regen_snapshots or not os.path.exists(path):
                np.savez(path, **arrays)
                return
            with np.load(path) as ref:
                assert set(ref.files) == set(arrays.keys()), (
                    f"Snapshot key mismatch for {name}: "
                    f"expected {sorted(ref.files)}, got {sorted(arrays.keys())}"
                )
                for key in ref.files:
                    np.testing.assert_allclose(
                        arrays[key],
                        ref[key],
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Mismatch in snapshot '{name}' key '{key}'",
                    )
        elif fmt == "json":
            path = os.path.join(SNAPSHOTS_DIR, f"{name}.json")
            if regen_snapshots or not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, sort_keys=True, default=str)
                return
            with open(path, encoding="utf-8") as fh:
                expected = json.load(fh)
            assert payload == expected, f"JSON snapshot '{name}' mismatch"
        elif fmt == "csv":
            path = os.path.join(SNAPSHOTS_DIR, f"{name}.csv")
            assert isinstance(payload, pd.DataFrame)
            if regen_snapshots or not os.path.exists(path):
                payload.to_csv(path, index=True)
                return
            expected = pd.read_csv(path, index_col=0)
            expected = expected[payload.columns]
            pd.testing.assert_frame_equal(
                payload.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False,
                rtol=rtol,
                atol=atol,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported snapshot format: {fmt}")

    return _compare

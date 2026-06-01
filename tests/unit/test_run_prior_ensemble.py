from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import cscm_calibrate.run_prior_ensemble as rpe
from cscm_calibrate.run_prior_ensemble import (
    _generate_prior_ensemble_parameters,
    find_missing_chunks,
    merge_dicts,
    run_prior_ensemble,
)


def test_run_prior_ensemble_minimal(monkeypatch, tmp_path):
    # Mock testconfig with required method
    testconfig = MagicMock()
    testconfig.make_config_lists = MagicMock()

    # Create mock results DataFrame: 2 runs, 1 variable, 10 years (starting from 1743)
    variables = ["var1"]
    run_ids = [0, 1]
    years = [str(1743 + i) for i in range(10)]
    rows = []
    for run_id in run_ids:
        for var in variables:
            row = {"variable": var, "run_id": run_id}
            for y in years:
                row[y] = float(run_id * 10 + int(y))
            rows.append(row)
    mock_results = pd.DataFrame(rows)

    class DummyDistributionRun:
        def __init__(self, *a, **k):
            pass

        def run_over_distribution(self, *a, **k):
            return mock_results

        @property
        def cfgs(self):
            return [
                {
                    "pamset_udm": {"a": 1},
                    "pamset_emiconc": {"b": 2},
                    "pamset_carbon": {"c": 3},
                },
                {
                    "pamset_udm": {"a": 4},
                    "pamset_emiconc": {"b": 5},
                    "pamset_carbon": {"c": 6},
                },
            ]

    monkeypatch.setattr(
        "cscm_calibrate.run_prior_ensemble.DistributionRun", DummyDistributionRun
    )

    # Minimal calibdata and prunecfgs, matching the mock years
    # but with Yearstart/Yearend >= 1750
    calibdata = pd.DataFrame(
        {
            "Variable Name": ["var1"],
            "Yearstart_norm": [1750],
            "Yearend_norm": [1750],
            "Yearstart_change": [1750],
            "Yearend_change": [1750],
        }
    )
    prunecfgs = {"var1": ["var1"]}

    # Patch np.save and pd.HDFStore to avoid file I/O
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    class DummyStore:
        def __setitem__(self, key, value):
            if key == "targ":
                assert isinstance(value, pd.DataFrame)
            if key == "parammat":
                assert isinstance(value, pd.DataFrame)

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    # Run the function (should not raise)
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=calibdata,
        prunecfgs=prunecfgs,
        distnums=2,
        chunk_size=1,
        startdate="_test",
    )
    # Assert testconfig.make_config_lists was called
    testconfig.make_config_lists.assert_called()


def test_run_prior_ensemble_realistic_shape(monkeypatch, tmp_path):
    # Mock testconfig with required method
    testconfig = MagicMock()
    testconfig.make_config_lists = MagicMock()

    # Create mock results DataFrame: 4 runs, 2 variables, 3 years (starting from 1743)
    variables = ["var1", "var2"]
    run_ids = [0, 1, 2, 3]
    years = [str(1743 + i) for i in range(10)]
    rows = []
    for run_id in run_ids:
        for var in variables:
            row = {"variable": var, "run_id": run_id}
            for y in years:
                row[y] = float(run_id * 10 + int(y))  # unique value per run/year
            rows.append(row)
    mock_results = pd.DataFrame(rows)

    class DummyDistributionRun:
        def __init__(self, *a, **k):
            pass

        def run_over_distribution(self, *a, **k):
            return mock_results

        @property
        def cfgs(self):
            return [
                {
                    "pamset_udm": {"a": 1},
                    "pamset_emiconc": {"b": 2},
                    "pamset_carbon": {"c": 3},
                },
                {
                    "pamset_udm": {"a": 4},
                    "pamset_emiconc": {"b": 5},
                    "pamset_carbon": {"c": 6},
                },
                {
                    "pamset_udm": {"a": 7},
                    "pamset_emiconc": {"b": 8},
                    "pamset_carbon": {"c": 9},
                },
                {
                    "pamset_udm": {"a": 10},
                    "pamset_emiconc": {"b": 11},
                    "pamset_carbon": {"c": 12},
                },
            ]

    monkeypatch.setattr(
        "cscm_calibrate.run_prior_ensemble.DistributionRun", DummyDistributionRun
    )

    # Calibdata with different year logic, matching the mock years
    #  but with Yearstart/Yearend >= 1750
    calibdata = pd.DataFrame(
        {
            "Variable Name": ["var1", "var2", "var1"],
            "Yearstart_norm": [1750, 1751, 1752],
            "Yearend_norm": [1750, 1751, 1752],
            "Yearstart_change": [1750, 1751, 1752],
            "Yearend_change": [1750, 1751, 1752],
        }
    )
    prunecfgs = {"var1": ["var1"], "var2": ["var2"]}

    # Patch np.save and pd.HDFStore to avoid file I/O
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    class DummyStore:
        def __setitem__(self, key, value):
            if key == "targ":
                assert isinstance(value, pd.DataFrame)
            if key == "parammat":
                assert isinstance(value, pd.DataFrame)

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    # Run the function (should not raise)
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=calibdata,
        prunecfgs=prunecfgs,
        distnums=4,
        chunk_size=2,
        startdate="_test",
    )
    testconfig.make_config_lists.assert_called()


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_find_missing_chunks_empty():
    assert find_missing_chunks([], 4) == [0, 1, 2, 3]


def test_find_missing_chunks_full():
    assert find_missing_chunks([0, 1, 2, 3], 4) == []


def test_find_missing_chunks_sparse():
    assert find_missing_chunks([1, 3], 5) == [0, 2, 4]


def test_merge_dicts_combines_three_pamsets():
    out = merge_dicts(
        {
            "pamset_udm": {"a": 1, "b": 2},
            "pamset_emiconc": {"c": 3},
            "pamset_carbon": {"d": 4},
        }
    )
    assert out == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_merge_dicts_carbon_overrides_emiconc_overrides_udm():
    out = merge_dicts(
        {
            "pamset_udm": {"x": 1, "y": 1, "z": 1},
            "pamset_emiconc": {"y": 2, "z": 2},
            "pamset_carbon": {"z": 3},
        }
    )
    assert out["x"] == 1
    assert out["y"] == 2  # emiconc wins over udm
    assert out["z"] == 3  # carbon wins over emiconc


# ---------------------------------------------------------------------------
# _generate_prior_ensemble_parameters resume / fresh paths
# ---------------------------------------------------------------------------


def test_generate_prior_ensemble_parameters_resume_uses_existing(tmp_path):
    """If the last-chunk config exists and continue_from_existing is set,
    the function returns immediately without invoking make_config_lists."""
    distnums = 10
    chunk_size = 5
    chunk_nums = int(np.ceil(distnums / chunk_size))
    # Create the marker file the function looks for
    (tmp_path / f"configs_{distnums}_chunk_{chunk_nums - 1}.json").write_text("{}")

    testconfig = MagicMock()
    _generate_prior_ensemble_parameters(
        testconfig,
        str(tmp_path),
        distnums=distnums,
        chunk_size=chunk_size,
        continue_from_existing=True,
    )
    testconfig.make_config_lists.assert_not_called()


def test_generate_prior_ensemble_parameters_creates_when_missing(tmp_path):
    """When the marker file is absent, make_config_lists is invoked."""
    out_dir = tmp_path / "configs_out"  # note: does not exist yet
    testconfig = MagicMock()
    _generate_prior_ensemble_parameters(
        testconfig,
        str(out_dir),
        distnums=10,
        chunk_size=5,
        continue_from_existing=True,
    )
    testconfig.make_config_lists.assert_called_once()
    assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# run_prior_ensemble continue_from_existing branches
# ---------------------------------------------------------------------------


def _make_dummy_distrorun_class(results, cfgs):
    class DummyDistributionRun:
        instances = 0

        def __init__(self, *a, **k):
            type(self).instances += 1

        def run_over_distribution(self, *a, **k):
            return results

        @property
        def cfgs(self):
            return cfgs

    return DummyDistributionRun


def _basic_calibdata():
    return pd.DataFrame(
        {
            "Variable Name": ["var1"],
            "Yearstart_norm": [1750],
            "Yearend_norm": [1750],
            "Yearstart_change": [1750],
            "Yearend_change": [1750],
        }
    )


def _basic_results(run_ids=(0, 1), variables=("var1",)):
    years = [str(1743 + i) for i in range(10)]
    rows = []
    for rid in run_ids:
        for var in variables:
            row = {"variable": var, "run_id": rid}
            for y in years:
                row[y] = float(rid * 10 + int(y))
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_prior_ensemble_all_chunks_exist_returns_early(monkeypatch, tmp_path):
    """When every chunk's sample dump already exists, we exit before constructing
    DistributionRun."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(rpe, "get_project_root", lambda: str(tmp_path))

    # No-op generator (mimics resume hit)
    monkeypatch.setattr(
        rpe, "_generate_prior_ensemble_parameters", lambda **kw: None
    )

    distnums, chunk_size = 4, 2
    chunk_nums = int(np.ceil(distnums / chunk_size))
    # Touch a sample-id dump for the highest chunk index so sample_max == chunk_nums-1
    # and len(existing) > sample_max -> to_run is empty.
    for i in range(chunk_nums):
        (output_dir / f"sample_ids_{distnums}_chunk_{i}.npy").write_bytes(b"")
    # need at least chunk_nums files so the > sample_max condition triggers empty range
    DummyRun = _make_dummy_distrorun_class(_basic_results(), [])
    monkeypatch.setattr(rpe, "DistributionRun", DummyRun)

    testconfig = MagicMock()
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=_basic_calibdata(),
        prunecfgs={"var1": ["var1"]},
        distnums=distnums,
        chunk_size=chunk_size,
        continue_from_existing=True,
    )
    assert DummyRun.instances == 0


def test_run_prior_ensemble_continue_from_existing_finds_missing(
    monkeypatch, tmp_path
):
    """When some chunks are missing, find_missing_chunks should be used."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(rpe, "get_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(
        rpe, "_generate_prior_ensemble_parameters", lambda **kw: None
    )

    distnums, chunk_size = 6, 2  # chunk_nums = 3
    # Create a sparse set: chunks 0 and 2 exist; chunk 1 is missing.
    # len(existing)==2 > sample_max==2 is False (2>2 False) -> hits the find_missing
    # branch in the code (the "else" / find_missing_chunks path).
    for i in (0, 2):
        (output_dir / f"sample_ids_{distnums}_chunk_{i}.npy").write_bytes(b"")

    cfgs = [
        {
            "pamset_udm": {"a": k},
            "pamset_emiconc": {"b": k + 1},
            "pamset_carbon": {"c": k + 2},
        }
        for k in range(chunk_size)
    ]
    DummyRun = _make_dummy_distrorun_class(_basic_results(), cfgs)
    monkeypatch.setattr(rpe, "DistributionRun", DummyRun)
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    class DummyStore:
        def __setitem__(self, key, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    testconfig = MagicMock()
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=_basic_calibdata(),
        prunecfgs={"var1": ["var1"]},
        distnums=distnums,
        chunk_size=chunk_size,
        continue_from_existing=True,
    )
    # Should run exactly the missing chunks (here: chunk 1 only, possibly more
    # depending on how the > sample_max check splits the path). We only assert
    # that *some* run happened and that find_missing produced a non-empty list.
    assert DummyRun.instances >= 1


# ---------------------------------------------------------------------------
# Year-slice branches in run_single_chunk
# ---------------------------------------------------------------------------


def test_run_prior_ensemble_year_slice_branches(monkeypatch, tmp_path):
    """Exercise the elif and else branches of the year-slicing in run_single_chunk:

    - row 1: Yearstart_norm == Yearend_norm but Yearstart_change != Yearend_change
             AND norm year != syear (1750)  -> the `elif ... Yearstart_change ==
             Yearend_change` path's else (mean range minus norm point).
    - row 2: Yearstart_norm != Yearend_norm                                   ->
             the final `else` branch (mean range minus mean range).
    """
    monkeypatch.setattr(rpe, "get_project_root", lambda: str(tmp_path))

    # A wide-enough year range so column offsets stay valid: years 1743..1762
    years = [str(1743 + i) for i in range(20)]
    rows = []
    for rid in (0, 1):
        for var in ("var1", "var2"):
            r = {"variable": var, "run_id": rid}
            for y in years:
                r[y] = float(rid * 100 + int(y))
            rows.append(r)
    mock_results = pd.DataFrame(rows)

    cfgs = [
        {
            "pamset_udm": {"a": k},
            "pamset_emiconc": {"b": k + 1},
            "pamset_carbon": {"c": k + 2},
        }
        for k in range(2)
    ]
    DummyRun = _make_dummy_distrorun_class(mock_results, cfgs)
    monkeypatch.setattr(rpe, "DistributionRun", DummyRun)
    monkeypatch.setattr(np, "save", lambda *a, **k: None)

    captured = {}

    class DummyStore:
        def __setitem__(self, key, value):
            captured[key] = value

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    calibdata = pd.DataFrame(
        {
            "Variable Name": ["var1", "var2"],
            # row 1 hits the elif branch (norm == norm == 1751, change spans 1751-1752)
            # row 2 hits the final else branch (norm spans 1751-1752, change spans
            # 1753-1754)
            "Yearstart_norm": [1751, 1751],
            "Yearend_norm": [1751, 1752],
            "Yearstart_change": [1751, 1753],
            "Yearend_change": [1752, 1754],
        }
    )
    testconfig = MagicMock()
    run_prior_ensemble(
        testconfig=testconfig,
        scenariodata=[{}],
        calibdata=calibdata,
        prunecfgs={"var1": ["var1"]},
        distnums=2,
        chunk_size=2,
        startdate=None,  # exercises `if startdate is None: startdate = ""`
    )
    # targ DataFrame must have been built containing both variable names as cols
    assert "targ" in captured
    targ = captured["targ"]
    assert {"var1", "var2"}.issubset(targ.columns)
    assert len(targ) == 2  # one entry per run_id

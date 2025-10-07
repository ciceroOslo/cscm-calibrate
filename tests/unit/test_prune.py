import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import cscm_calibrate.prune_distribution_to_timeseries as prune


def test_prepare_weights_temp(monkeypatch, tmp_path):
    # Create a dummy CSV file
    df = pd.DataFrame({"GMST": np.arange(52)})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    gmst, weights = prune.prepare_weights_temp(str(csv_path))
    assert np.allclose(gmst, np.arange(52))
    assert weights.shape == (52,)
    assert weights[0] == 0.5
    assert weights[-1] == 0.0


def test_do_pruning_for_chunk(monkeypatch, tmp_path):
    # Patch prepare_weights_temp to return known gmst and weights
    monkeypatch.setattr(
        prune, "prepare_weights_temp", lambda path: (np.ones(173), np.ones(52))
    )
    # Patch np.load to return dummy samples and temp_in
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **k: (
            np.ones((2, 174)) if "1850-2023" in a[0] else np.array([10, 20])
        ),
    )
    # Patch rmse to return a fixed value
    monkeypatch.setattr(prune, "rmse", lambda a, b: 0.5)
    # Should accept all samples if rmse_accept > 0.5
    valid_temp, accept_temp, samples = prune.do_pruning_for_chunk(
        0, ["foo", "bar", 1.0], file_endstring="", total_samples=2
    )
    assert np.all(accept_temp)
    assert len(valid_temp) == 2
    assert np.all(samples == np.array([10, 20]))


def test_get_targ_paramat_valid_for_chunk(monkeypatch, tmp_path):
    # Patch pd.HDFStore to return dummy DataFrames
    class DummyStore:
        def __getitem__(self, key):
            return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())
    targ, parammat = prune.get_targ_paramat_valid_for_chunk(0, [0, 2])
    assert targ.shape[0] == 2
    assert parammat.shape[0] == 2


def test_prune_all_chunks(monkeypatch):
    # Patch do_pruning_for_chunk and get_targ_paramat_valid_for_chunk
    monkeypatch.setattr(
        prune,
        "do_pruning_for_chunk",
        lambda chunk_num, prune_list, file_endstring, total_samples: (
            np.array([0, 1]),
            np.array([True, True]),
            np.array([10, 20]),
        ),
    )
    monkeypatch.setattr(
        prune,
        "get_targ_paramat_valid_for_chunk",
        lambda chunk_num, valid_samples, total_samples: (
            pd.DataFrame({"A": [1, 2]}),
            pd.DataFrame({"B": [3, 4]}),
        ),
    )
    monkeypatch.setattr(np, "save", lambda *a, **k: None)
    monkeypatch.setattr(
        pd,
        "concat",
        lambda dfs: pd.DataFrame(
            {k: np.concatenate([df[k].values for df in dfs]) for k in dfs[0].columns}
        ),
    )

    class DummyStore:
        def __setitem__(self, key, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())
    prune.prune_all_chunks(
        2, [["foo", "bar", 1.0]], num_chunks=2, file_endstring="_test"
    )
    # If no exception, test passes

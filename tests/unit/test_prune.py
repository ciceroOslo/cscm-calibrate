import numpy as np
import pandas as pd
import pytest

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

        def keys(self):
            return ["/targ", "/parammat"]

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
        lambda chunk_num, valid_samples, total_samples, file_endstring: (
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


# ---------------------------------------------------------------------------
# Class-based tests that share real on-disk fixture files via self.
# ---------------------------------------------------------------------------


@pytest.fixture
def pruning_fixtures(tmp_path, monkeypatch):
    """Build the npy / csv files do_pruning_for_chunk expects."""
    project_root = tmp_path
    output_dir = project_root / "output"
    output_dir.mkdir()
    monkeypatch.setattr(prune, "get_project_root", lambda: str(project_root))

    # 173 GMST values (matches the gmst[:173] slice the function uses)
    gmst = np.arange(173, dtype=float) * 0.01
    gmst_csv = tmp_path / "gmst.csv"
    pd.DataFrame({"GMST": np.concatenate([gmst, [0.0]])}).to_csv(
        gmst_csv, index=False
    )

    # Three samples; baseline (cols 0..51) starts as zeros for everyone.
    # Members 0 & 1 follow gmst closely after baseline subtraction;
    # member 2's temperature has the *opposite* trend so it cannot be
    # rescued by baseline subtraction and will fail the rmse threshold.
    n_samples = 3
    temp_in = np.zeros((n_samples, 174))
    for i in range(n_samples):
        if i == 2:
            temp_in[i, 1:] = -gmst  # reversed trend: large rmse
        else:
            temp_in[i, 1:] = gmst + 0.001 * i
    total = 100
    chunk_num = 0
    fname = (
        output_dir / f"foo_{total}_chunk_{chunk_num}_1850-2023.npy"
    )
    np.save(fname, temp_in)
    sample_ids = np.array([10, 20, 30])
    np.save(
        output_dir / f"sample_ids_{total}_chunk_{chunk_num}.npy", sample_ids
    )

    # HDF5 mock for get_targ_paramat_valid_for_chunk
    targ_df = pd.DataFrame({"target": np.arange(n_samples, dtype=float)})
    pmat_df = pd.DataFrame({"param": np.arange(n_samples, dtype=float) * 2})

    class DummyStore:
        def __init__(self, *a, **k):
            pass

        def __getitem__(self, key):
            return targ_df if key == "targ" else pmat_df

        def keys(self):
            return ["/targ", "/parammat"]

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", DummyStore)

    return {
        "gmst_csv": str(gmst_csv),
        "total": total,
        "chunk_num": chunk_num,
        "sample_ids": sample_ids,
        "n_samples": n_samples,
        "output_dir": output_dir,
    }


class TestPruningPipeline:
    """Real per-fixture exercise of the pruning helpers."""

    @pytest.fixture(autouse=True)
    def _setup(self, pruning_fixtures):
        self.fix = pruning_fixtures

    def test_do_pruning_filters_by_rmse_threshold(self):
        """Members 0 & 1 below threshold pass; member 2 (offset 5) is rejected."""
        valid_idx, accept_mask, kept_ids = prune.do_pruning_for_chunk(
            self.fix["chunk_num"],
            ["foo", self.fix["gmst_csv"], 1.0],
            file_endstring=None,  # exercises the None -> "" branch
            total_samples=self.fix["total"],
        )
        assert accept_mask.tolist() == [True, True, False]
        assert valid_idx.tolist() == [0, 1]
        assert kept_ids.tolist() == [10, 20]

    def test_do_pruning_with_explicit_endstring(self, monkeypatch):
        """When file_endstring is provided it must form part of the filename."""
        # Re-save the temp_in npy file under the suffix-augmented name
        temp_in = np.load(
            self.fix["output_dir"]
            / f"foo_{self.fix['total']}_chunk_{self.fix['chunk_num']}_1850-2023.npy"
        )
        suffix = "_v2"
        np.save(
            self.fix["output_dir"]
            / f"foo_{self.fix['total']}_chunk_{self.fix['chunk_num']}{suffix}_1850-2023.npy",
            temp_in,
        )
        np.save(
            self.fix["output_dir"]
            / f"sample_ids_{self.fix['total']}_chunk_{self.fix['chunk_num']}{suffix}.npy",
            self.fix["sample_ids"],
        )
        valid_idx, _, _ = prune.do_pruning_for_chunk(
            self.fix["chunk_num"],
            ["foo", self.fix["gmst_csv"], 0.5],
            file_endstring=suffix,
            total_samples=self.fix["total"],
        )
        # All within the lower threshold pass, the rmse=5 member still rejected
        assert len(valid_idx) == 2

    def test_get_targ_paramat_valid_for_chunk_with_none_endstring(self):
        """file_endstring=None branch must convert to empty string and succeed."""
        targ, parammat = prune.get_targ_paramat_valid_for_chunk(
            self.fix["chunk_num"],
            [0, 2],
            total_samples=self.fix["total"],
            file_endstring=None,
        )
        assert targ.shape[0] == 2
        assert parammat.shape[0] == 2
        # iloc selects rows 0 and 2 (not 0..1)
        assert targ["target"].tolist() == [0.0, 2.0]


def test_prune_all_chunks_warns_on_multiple_lists(monkeypatch, capsys):
    """The multi-prune-list branch prints a TODO message and proceeds with the first."""
    monkeypatch.setattr(
        prune,
        "do_pruning_for_chunk",
        lambda chunk_num, prune_list, file_endstring, total_samples: (
            np.array([0]),
            np.array([True]),
            np.array([99]),
        ),
    )
    monkeypatch.setattr(
        prune,
        "get_targ_paramat_valid_for_chunk",
        lambda chunk_num, valid_samples, total_samples, file_endstring: (
            pd.DataFrame({"A": [1]}),
            pd.DataFrame({"B": [2]}),
        ),
    )
    monkeypatch.setattr(np, "save", lambda *a, **k: None)
    monkeypatch.setattr(prune, "get_project_root", lambda: "/tmp")

    class DummyStore:
        def __setitem__(self, key, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pd, "HDFStore", lambda *a, **k: DummyStore())

    prune.prune_all_chunks(
        2,
        [["foo", "bar", 1.0], ["baz", "qux", 0.5]],
        num_chunks=1,
        file_endstring="_x",
    )
    out = capsys.readouterr().out
    assert "Currently unimplemented" in out
    # If no exception, test passes

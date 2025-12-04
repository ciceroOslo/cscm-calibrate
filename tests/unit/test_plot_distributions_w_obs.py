import numpy as np
import pandas as pd

import cscm_calibrate.plot_distributions_w_obs as pdwo


def test_read_noaa_gml_ml_means_reads_csv(monkeypatch):
    df = pd.DataFrame(
        {
            "year": [2000, 2001],
            "mean": [1.0, 2.0],
            "annmean": [1.0, 2.0],
            "decimal": [1.0, 2.0],
            "average": [1.0, 2.0],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: df)
    arr = pdwo.read_noaa_gml_ml_means("year")
    assert arr.shape[0] == 2


def test_read_gcb_data(monkeypatch):
    df = pd.DataFrame(
        {
            "fossil emissions excluding carbonation": [1.0],
            "land-use change emissions": [2.0],
            "cement carbonation sink": [0.5],
        }
    )
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df.copy())
    result = pdwo.read_gcb_data()
    assert "emissions_tot" in result.columns
    assert "fossil emissions" in result.columns
    assert np.isclose(result["emissions_tot"].iloc[0], 3.5)
    assert np.isclose(result["fossil emissions"].iloc[0], 1.5)


def test_read_gcb_ocean_carbon_data(monkeypatch):
    df = pd.DataFrame({"A": [1], "B": [2], "Unnamed: 0": [0]})
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df)
    result = pdwo.read_gcb_ocean_carbon_data()
    assert "Unnamed: 0" not in result.columns


def test_pam_plotting_creates_figure(monkeypatch):
    class DummyFig:
        def suptitle(self, *a, **k):
            pass

        def savefig(self, *a, **k):
            self.saved = True

    class DummyAx:
        def hist(self, *a, **k):
            pass

        def set_title(self, *a, **k):
            pass

    monkeypatch.setattr(
        pdwo.plt,
        "subplots",
        lambda nrows, ncols, figsize: (DummyFig(), np.array([[DummyAx()] * 5] * 5)),
    )
    monkeypatch.setattr(pdwo.plt, "clf", lambda: None)
    parammat = pd.DataFrame({f"p{i}": np.arange(10) for i in range(25)})
    pdwo.pam_plotting(parammat)


def test_get_data_for_plots(monkeypatch):
    monkeypatch.setattr(
        pd, "read_csv", lambda *a, **k: pd.DataFrame({"time": [2000], "GMST": [1.0]})
    )
    monkeypatch.setattr(
        pdwo, "read_noaa_gml_ml_means", lambda timeres: np.array([[2000], [400.0]])
    )
    monkeypatch.setattr(
        pdwo,
        "read_gcb_data",
        lambda: pd.DataFrame({"Year": [2000], "ocean sink": [1.0], "land sink": [2.0]}),
    )
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame({"A": [1]}))
    result = pdwo.get_data_for_plots()
    assert len(result) == 7


def test_plot_distributions_calls(monkeypatch):
    dummy_results = pd.DataFrame(
        np.random.rand(2, 8),
        columns=[f"c{i}" for i in range(5)] + [str(2000 + i) for i in range(3)],
    )

    dummy_results["variable"] = [
        "Heat Content|Ocean",
        "Surface Air Ocean Blended Temperature Change",
    ]
    dummy_results["variable"] = dummy_results["variable"].astype(str)
    dummy_results["units"] = ["ZJ", "K"]

    cols = dummy_results.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    dummy_results = dummy_results[cols]
    monkeypatch.setattr(
        pdwo,
        "get_data_for_plots",
        lambda: (
            pd.DataFrame({"time": [2000], "GMST": [1.0]}),
            np.array([[2000], [400.0]]),
            pd.DataFrame({"Year": [2000], "ocean sink": [1.0], "land sink": [2.0]}),
            pd.DataFrame(
                {
                    "time": [2000],
                    "aerosol-radiation_interactions": [0.1],
                    "aerosol-cloud_interactions": [0.2],
                }
            ),
            pd.DataFrame(
                {
                    "aerosol-radiation_interactions": [0.1],
                    "aerosol-cloud_interactions": [0.2],
                }
            ),
            pd.DataFrame(
                {
                    "aerosol-radiation_interactions": [0.1],
                    "aerosol-cloud_interactions": [0.2],
                }
            ),
            pd.DataFrame(
                {
                    "Year": [2000],
                    "Central Estimate Full-depth": [1.0],
                    "Full-depth Uncertainty (1-sigma)": [0.1],
                }
            ),
        ),
    )
    monkeypatch.setattr(pdwo.plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(pdwo.plt, "clf", lambda: None)
    print(dummy_results)
    pdwo.plot_distributions(dummy_results, "test")

import numpy as np
import pandas as pd
import pytest

from cscm_calibrate.esm.targets import HIST_END, idealised_blocks, scenario_blocks

SRC = "ESM-X"
YEARS = np.arange(1850, 2101)
SIGMA = 0.1  # single-member noise of the synthetic runs


@pytest.fixture
def data_dir(tmp_path):
    """Minimal synthetic products: 4 members, tiers H and VL, shared history."""
    rng = np.random.default_rng(0)
    rows = []
    for v in range(4):
        hist = SIGMA * rng.standard_normal(len(YEARS))  # shared by both tiers
        for tier, slope in (("H", 0.04), ("VL", 0.0)):
            noise = np.where(
                YEARS > HIST_END, SIGMA * rng.standard_normal(len(YEARS)), hist
            )
            fut = (
                0.01 * (YEARS - 1850)
                + slope * np.clip(YEARS - HIST_END, 0, None)
                + noise
            )
            rows += [
                ("ESM", SRC, f"r{v}", "concentration", tier, y, t, 0.0, 0.0)
                for y, t in zip(YEARS, fut)
            ]
    cols = ["ensemble", "source", "variant", "driver", "tier", "year", "gsat_raw_K"]
    g = pd.DataFrame(rows, columns=[*cols, "net_toa_Wm2", "net_toa_control_Wm2"])
    g[cols].to_csv(tmp_path / "gsat.csv", index=False)
    g.to_csv(tmp_path / "toa_imbalance.csv", index=False)
    pd.DataFrame(
        {
            "source": SRC,
            "experiment": "piControl",
            "year": np.arange(500),
            "tas_K": np.nan,
            "N_Wm2": 0.3 * rng.standard_normal(500),
        }
    ).to_csv(tmp_path / "energy_closure.csv", index=False)
    i = np.arange(1, 201)
    pd.DataFrame(
        {
            "source": SRC,
            "year_index": i,
            "dT_K": 6 * (1 - np.exp(-i / 20)),
            "dN_Wm2": 7 * np.exp(-i / 20),
        }
    ).to_csv(tmp_path / "gregory.csv", index=False)
    pd.DataFrame({"source": SRC, "year_index": i[:140], "dT_K": 0.02 * i[:140]}).to_csv(
        tmp_path / "transient_1pct.csv", index=False
    )
    return tmp_path


def test_scenario_blocks_split_history_and_tiers(data_dir):
    b = scenario_blocks(data_dir, SRC, "concentration")
    assert set(b) == {f"{t}/{v}" for t in ("hist", "H", "VL") for v in "TN"}
    assert b["hist/T"].x[-1] == HIST_END
    assert b["H/T"].x[0] == HIST_END + 1
    assert b["hist/T"].k == 4  # history counted once, not once per tier
    assert b["H/T"].y[-1] - b["VL/T"].y[-1] == pytest.approx(0.04 * 79, abs=0.2)


def test_missing_variable_is_skipped(data_dir):
    toa = pd.read_csv(data_dir / "toa_imbalance.csv")
    toa.assign(net_toa_Wm2=np.nan).to_csv(data_dir / "toa_imbalance.csv", index=False)
    b = scenario_blocks(data_dir, SRC, "concentration")
    assert set(b) == {f"{t}/T" for t in ("hist", "H", "VL")}


def test_noise_from_spread_or_control(data_dir):
    b = scenario_blocks(data_dir, SRC, "concentration")
    # piControl has no tas: T noise comes from the member spread
    assert b["H/T"].sigma * np.sqrt(4) == pytest.approx(SIGMA, rel=0.15)
    # piControl N is available and preferred
    assert b["H/N"].sigma * np.sqrt(4) == pytest.approx(0.3, rel=0.15)


def test_idealised_blocks(data_dir):
    b = idealised_blocks(data_dir, SRC)
    assert len(b["abrupt-4xCO2/T"].x) == 150
    assert b["1pctCO2/T"].x[-1] == 140

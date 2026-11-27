"""Check: forcing-driven rebuild with unit scalings == standard conc-driven CICERO-SCM run.

Only meaningful for the 1750 start (the 1850 start removes the constant pre-1850 forcing).
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ["CMIP7_BASELINE"] = "1750"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulator as EM  # noqa: E402
import scm_forcing as SF  # noqa: E402

th = EM.default_theta()
em = EM.Emulator(["H", "VL"], idealised=False)
for tier in ("H", "VL"):
    m = SF._CSCM(SF.scenario_scendata(tier))
    m._run({"results_as_dict": True}, pamset_udm=EM.pamset_udm(th))
    ref = m.results
    o = em.run(th, tiers=[tier], idealised=False)
    T_ref = ref[EM.T_VAR] - ref[EM.T_VAR][em.base].mean()
    dT = np.abs(o[f"{tier}/T"][1] - T_ref).max()
    dN = np.abs(o[f"{tier}/N"][1] - ref["RIB_glob"]).max()
    i = 2019 - SF.NYSTART
    print(f"{tier}: max |dT| {dT:.2e} K, max |dN| {dN:.2e} W/m2; ref T2019 {T_ref[i]:.2f} T2100 {T_ref[-1]:.2f}; "
          f"volc mean 1850-2019 {ref['Volcanic_forcing_NH'][100:270].mean():.2f}")

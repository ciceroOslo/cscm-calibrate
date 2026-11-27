"""Checks for the emissions-driven stage.

1. 1750 baseline: EExp with unit scalings == plain emissions-driven CICERO-SCM.
2. 1850 (pic) baseline: pre-1850 state is constant (CO2, T, N, ocean flux ~ 0 at 1850),
   and the stage-1 forcing-driven emulator agrees with it before 1850.
3. Time one log_prob with the full carbon parameter set.
"""
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["CMIP7_BASELINE"] = "1750"
import scm_forcing as SF  # noqa: E402
import emulator as EM  # noqa: E402
import fit_carbon as FC  # noqa: E402

th = EM.default_theta()
carbon = {"beta_f": 0.4, "mixed_carbon": 100.0}
e = FC.EExp("H", SF.load())
o = e.run(th, carbon)
sd = SF.scenario_scendata("H")
sd.update(conc_run=False, emstart=1850)
m = SF._CSCM(sd)
m._run({"results_as_dict": True, "carbon_cycle_outputs": True}, pamset_udm=EM.pamset_udm(th),
       pamset_carbon={**FC.CARBON_SET, **carbon})
T = m.results[EM.T_VAR] - m.results[EM.T_VAR][e.base].mean()
print("1750 baseline: max |dT|", np.abs(o["T"] - T).max(),
      "max |dCO2|", np.abs(o["C"] - m.results["concentrations"]["CO2"].to_numpy()).max())

os.environ["CMIP7_BASELINE"] = "pic"
for mod in (SF, EM, FC):
    importlib.reload(mod)
comps = SF.load()
e = FC.EExp("H", comps)
o = e.run(th, carbon)
i0 = 1850 - SF.NYSTART
yr = lambda y: y - SF.NYSTART  # noqa: E731
print("pic baseline: CO2 1750/1849/1850/1900/2021:", [round(o["C"][yr(y)], 2) for y in (1750, 1849, 1850, 1900, 2021)])
print("  ocean flux 1849/1850/1860/1900:", [round(o["O"][yr(y)], 3) for y in (1849, 1850, 1860, 1900)])
print("  land flux 1849/1850/1900:", [round(o["L"][yr(y)], 3) for y in (1849, 1850, 1900)])
print("  T range before 1850:", round(o["T"][:i0].min(), 4), round(o["T"][:i0].max(), 4))
em = EM.Emulator(["H"], idealised=False)
s1 = em.run(th, tiers=["H"], idealised=False)
print("  stage-1 emulator T/N before 1850 (max abs):", round(np.abs(s1["H/T"][1][:i0]).max(), 4),
      round(np.abs(s1["H/N"][1][:i0]).max(), 4), "N 1850-1900 mean", round(s1["H/N"][1][i0:i0 + 51].mean(), 3))
FC._init("UKESM1-3-LL", "eps", None, ["H", "M", "VL"], theta1=th)
c = 0.5 * (FC.CLO + FC.CHI)
t = time.time()
print("log_prob", FC.log_prob(c), f"{time.time() - t:.2f}s")

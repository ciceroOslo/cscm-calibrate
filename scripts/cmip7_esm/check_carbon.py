"""Check: EExp with unit scalings == plain emissions-driven CICERO-SCM; time one log_prob."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulator as EM  # noqa: E402
import fit_carbon as FC  # noqa: E402
import scm_forcing as SF  # noqa: E402

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
print("max |dT|", np.abs(o["T"] - T).max(), "max |dCO2|", np.abs(o["C"] - m.results["concentrations"]["CO2"].to_numpy()).max())
FC._init("UKESM1-3-LL", "eps", None, ["H", "M", "VL"], theta1=th)
c = 0.5 * (FC.CLO + FC.CHI)
t = time.time()
print("log_prob", FC.log_prob(c), f"{time.time() - t:.2f}s")

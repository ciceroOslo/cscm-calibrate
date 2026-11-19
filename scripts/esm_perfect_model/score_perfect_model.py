"""Score SCM posterior projections against the ESM's own scenario runs (perfect-model test).

For each SSP and window, compares the SCM posterior distribution of surface air temperature
change with the ESM members (same physics/forcing variant as the truth member):
  * percentiles of the SCM posterior (5, 17, 50, 83, 95)
  * truth member and ESM ensemble mean: rank in the SCM distribution (PIT) and whether they
    fall inside the 5-95 and 17-83 ranges
Also reports the historical fit (SAT 2004-2023 vs 1850-1900) against the pseudo-target.

Outputs (output/esm_perfect_model/<MODEL>/):
  score{suffix}.csv   one row per scenario x window
  score{suffix}.png   plumes 1850-2100 with ESM members, and end-of-century ranges

Usage: .venv-scm/bin/python scripts/esm_perfect_model/score_perfect_model.py MODEL [--fixed-skew] [--hist-only]
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VARIANT = os.environ.get("PM_VARIANT", "ar7")
RUNS = Path(os.environ.get("PM_RUNS", ROOT / "output" / ("esm_perfect_model" + ("" if VARIANT == "ar7" else f"_{VARIANT}"))))
TARGETS = ROOT / "data" / "esm_perfect_model" / "targets"
SAT = "Surface Air Temperature Change"
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
WINDOWS = {  # name: (period, baseline)
    "2081-2100 vs 1995-2014": ((2081, 2100), (1995, 2014)),
    "2041-2060 vs 1995-2014": ((2041, 2060), (1995, 2014)),
    "2081-2100 vs 1850-1900": ((2081, 2100), (1850, 1900)),
}
Q = [5, 17, 50, 83, 95]
SCM_C, ESM_C, TRUTH_C = "#2a6fbb", "#7a7a7a", "#1a1a1a"


def variant(m):
    return re.sub(r"^r\d+", "", m)  # i1p1f1


def change(df, period, base):
    """df: rows = members, columns = int years."""
    return df.loc[:, period[0]:period[1]].mean(1) - df.loc[:, base[0]:base[1]].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--fixed-skew", action="store_true")
    ap.add_argument("--hist-only", action="store_true")
    a = ap.parse_args()
    sfx = ("_histonly" if a.hist_only else "") + ("_fixedskew" if a.fixed_skew else "")
    label = (("history only" if a.hist_only else "with ECS/TCR") + ", "
             + ("skew-normal fix" if a.fixed_skew else "weighting as in AR7 pipeline"))
    run = RUNS / a.model

    ser = pd.read_csv(TARGETS / a.model / "series.csv")
    targ = pd.read_csv(TARGETS / a.model / "targets.csv")
    truth = targ.member.iloc[0]
    members = [m for m in ser.member.unique() if variant(m) == variant(truth)]
    ser = ser[ser.member.isin(members)]

    scm, esm = {}, {}
    for scen in SSPS:
        d = pd.read_csv(run / f"projections{sfx}" / f"{scen}.csv")
        d = d[d.variable == SAT].set_index("run_id")
        yc = [c for c in d.columns if c.isdigit()]
        d = d[yc]
        d.columns = d.columns.astype(int)
        scm[scen] = d
        e = ser[ser.scenario == scen].pivot(index="member", columns="year", values="tas")
        esm[scen] = e

    rows = []
    for scen in SSPS:
        if esm[scen].empty:
            continue
        for wname, (per, base) in WINDOWS.items():
            s = change(scm[scen], per, base).values
            e = change(esm[scen], per, base)
            row = dict(scenario=scen, window=wname, n_scm=len(s), n_esm=len(e),
                       **{f"scm_p{q}": np.percentile(s, q) for q in Q})
            for lab, v in [("truth", e.get(truth, np.nan)), ("esm_mean", e.mean())]:
                row[lab] = v
                row[f"{lab}_pit"] = (s < v).mean()
                row[f"{lab}_in_5_95"] = row["scm_p5"] <= v <= row["scm_p95"]
                row[f"{lab}_in_17_83"] = row["scm_p17"] <= v <= row["scm_p83"]
            row["esm_min"], row["esm_max"] = e.min(), e.max()
            row["esm_members_in_5_95"] = f"{((e >= row['scm_p5']) & (e <= row['scm_p95'])).sum()}/{len(e)}"
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(run / f"score{sfx}.csv", index=False)

    h = change(scm["ssp245"], (2004, 2023), (1850, 1900))
    print(f"{a.model} ({label}); truth {truth}; ESM members {members}")
    print(f"historical SAT 2004-2023: SCM {np.percentile(h, 50):.2f} [{np.percentile(h, 5):.2f}, "
          f"{np.percentile(h, 95):.2f}]  vs pseudo-target {targ.GMST_2004_2023.iloc[0]:.2f}")
    pd.set_option("display.width", 220)
    cols = ["scenario", "window", "scm_p5", "scm_p50", "scm_p95", "truth", "truth_pit", "esm_mean",
            "esm_mean_pit", "esm_members_in_5_95"]
    print(S[cols].round(2).to_string(index=False))

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9], wspace=0.28, hspace=0.32)
    for i, scen in enumerate(SSPS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        d = scm[scen].loc[:, 1850:2100]
        d = d.sub(d.loc[:, 1850:1900].mean(1), axis=0)
        p = np.percentile(d.values, Q, axis=0)
        yrs = d.columns.values
        ax.fill_between(yrs, p[0], p[4], color=SCM_C, alpha=0.15, lw=0, label="SCM 5–95%")
        ax.fill_between(yrs, p[1], p[3], color=SCM_C, alpha=0.3, lw=0, label="SCM 17–83%")
        ax.plot(yrs, p[2], color=SCM_C, lw=1.8, label="SCM median")
        for m, r in esm[scen].iterrows():
            ax.plot(r.index, r.values, color=TRUTH_C if m == truth else ESM_C,
                    lw=1.0 if m == truth else 0.6, alpha=1 if m == truth else 0.7,
                    label=f"{a.model} {m} (truth)" if m == truth else None, zorder=3 if m == truth else 2)
        ax.set_title(scen.upper().replace("SSP", "SSP"), fontsize=9, loc="left")
        ax.set_xlim(1850, 2100)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i % 2 == 0:
            ax.set_ylabel("SAT change vs 1850–1900 (K)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    bx = fig.add_subplot(gs[:, 2])
    w = S[S.window == "2081-2100 vs 1995-2014"].reset_index(drop=True)
    for j, r in w.iterrows():
        bx.plot([j, j], [r.scm_p5, r.scm_p95], color=SCM_C, lw=6, alpha=0.3, solid_capstyle="butt")
        bx.plot([j, j], [r.scm_p17, r.scm_p83], color=SCM_C, lw=6, alpha=0.6, solid_capstyle="butt")
        bx.plot([j - 0.18, j + 0.18], [r.scm_p50] * 2, color=SCM_C, lw=1.5)
        e = change(esm[r.scenario], (2081, 2100), (1995, 2014))
        bx.scatter(np.full(len(e), j + 0.28), e.values, s=12, color=ESM_C, zorder=3)
        bx.scatter([j + 0.28], [e[truth]], s=22, color=TRUTH_C, zorder=4)
    bx.set_xticks(range(len(w)), [s.upper() for s in w.scenario], fontsize=7)
    bx.set_ylabel("SAT change 2081–2100 vs 1995–2014 (K)", fontsize=8)
    bx.tick_params(labelsize=7)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.set_title("SCM 5–95 / 17–83 % and median; dots = ESM members (black = truth)", fontsize=7, loc="left")
    fig.suptitle(f"Perfect-model test: CICERO-SCM calibrated to {a.model} {truth} "
                 f"({label}), "
                 f"{len(scm['ssp245'])} posterior members", fontsize=9, x=0.01, ha="left")
    fig.savefig(run / f"score{sfx}.png", dpi=150, bbox_inches="tight")
    print("wrote", run / f"score{sfx}.png")


if __name__ == "__main__":
    main()

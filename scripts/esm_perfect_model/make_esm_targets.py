"""Drift-correct ESM global means and build perfect-model pseudo-observations and targets.

For each model in data/esm_perfect_model/global_means/:
  * drift: for every historical member, find the piControl year it branched from
    (branch_time_in_parent, parent_time_units, piControl calendar) and fit a straight line to
    piControl tas and N over the years parallel to historical + scenario (1850-2100).
    tas: subtract the fitted trend (anomalies are taken against 1850-1900 anyway).
    N:   subtract the full fit (removes the control-run TOA offset as well as drift).
    SSP runs use the branch of the historical member they continue.
  * series: drift-corrected tas anomaly vs the member's own 1850-1900 mean, and N, for
    historical spliced with each SSP (historical up to 2014, SSP from 2015).
  * targets: per member, as the AR7 calibration defines them:
      GMST  mean 2004-2023 minus mean 1850-1900 (historical + ssp245)
      OHC   total heat uptake from mid-1971 to mid-2020 = trapezoid integral of N, in ZJ
            (the AR7 target is total Earth heat uptake and is compared with the SCM's
            Heat Content|Ocean; the pseudo-target is treated the same way)

Outputs (data/esm_perfect_model/targets/<model>/):
  series.csv        member, scenario, year, tas, N            (all members)
  pseudo_obs.csv    year, GMST                                (truth member; the pruning series)
  targets.csv       member, GMST_2004_2023, OHC_1971_2020_ZJ  (all members)
  drift.csv         member, branch_year, window, slopes and offsets used

Usage: .venv/bin/python scripts/esm_perfect_model/make_esm_targets.py [model ...] [--truth r1i1p1f1]
"""

import argparse
import re
from pathlib import Path

import cftime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GM = ROOT / "data" / "esm_perfect_model" / "global_means"
OUTROOT = ROOT / "data" / "esm_perfect_model" / "targets"
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
EARTH_AREA = 5.1e14  # m2, as ciceroscm
ZJ_PER_WM2_YR = EARTH_AREA * 365.25 * 86400 / 1e21
SPAN = 2100 - 1850 + 1  # piControl years parallel to historical + scenario


def load(model):
    d = pd.read_csv(GM / f"{model}.csv")
    meta = pd.read_csv(GM / f"{model}_meta.csv")
    p = d.pivot_table(index=["experiment_id", "member_id", "year"], columns="variable", values="value")
    p["N"] = p["rsdt"] - p["rsut"] - p["rlut"]
    return p[["tas", "N"]], meta


def branch_year(meta, member, pic_years, pic_cal):
    m = meta[(meta.experiment_id == "historical") & (meta.member_id == member)].iloc[0]
    try:
        y = cftime.num2date(float(m.branch_time_in_parent), m.parent_time_units, calendar=pic_cal).year
    except Exception:
        y = None
    ok = y is not None and pic_years.min() <= y <= pic_years.max()
    return (y if ok else int(pic_years.min())), ok, m.parent_variant_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--truth", default=None, help="member used as pseudo-observations (default: first)")
    a = ap.parse_args()
    models = a.models or sorted(f.stem for f in GM.glob("*.csv") if not f.stem.endswith("_meta"))

    for model in models:
        p, meta = load(model)
        exps = set(p.index.get_level_values(0))
        if not {"historical", "piControl", "ssp245"} <= exps:
            print(f"{model}: missing historical/piControl/ssp245, skipped")
            continue
        pic = p.loc["piControl"].droplevel(0)
        pic_cal = meta.loc[meta.experiment_id == "piControl", "calendar"].iloc[0]
        pic_member = meta.loc[meta.experiment_id == "piControl", "member_id"].iloc[0]
        rows, drift, targ = [], [], []
        for mem in sorted(set(p.loc["historical"].index.get_level_values(0)),
                          key=lambda m: tuple(int(x) for x in re.findall(r"\d+", m))):
            b, ok, parent = branch_year(meta, mem, pic.index.values, pic_cal)
            win = pic.loc[b : b + SPAN - 1]
            if len(win) < 100:  # short control: use all of it
                win = pic
            t = win.index.values - b  # years since branch
            ft = np.polyfit(t, win.tas.values, 1)
            fn = np.polyfit(t, win.N.values, 1)
            drift.append(dict(member=mem, branch_year=b, branch_ok=ok, parent_variant_label=parent,
                              piControl_member=pic_member, window=f"{win.index.min()}-{win.index.max()}",
                              tas_slope_K_per_century=ft[0] * 100, N_offset=fn[1],
                              N_slope_per_century=fn[0] * 100))
            h = p.loc[("historical", mem)]
            for scen in SSPS:
                if (scen, mem) not in p.index.droplevel(2):
                    continue
                s = pd.concat([h.loc[:2014], p.loc[(scen, mem)].loc[2015:]])
                dt = s.index.values - 1850
                s = s.assign(tas=s.tas - np.polyval(ft, dt) + ft[1], N=s.N - np.polyval(fn, dt))
                s["tas"] -= s.tas.loc[1850:1900].mean()
                rows.append(s.reset_index().assign(member=mem, scenario=scen))
                if scen == "ssp245":
                    w = s.N.loc[1971:2020].values
                    ohc = (w.sum() - 0.5 * (w[0] + w[-1])) * ZJ_PER_WM2_YR
                    targ.append(dict(member=mem, GMST_2004_2023=s.tas.loc[2004:2023].mean(),
                                     OHC_1971_2020_ZJ=ohc, last_year=s.index.max()))
        out = OUTROOT / model
        out.mkdir(parents=True, exist_ok=True)
        ser = pd.concat(rows)[["member", "scenario", "year", "tas", "N"]]
        ser.to_csv(out / "series.csv", index=False)
        pd.DataFrame(drift).to_csv(out / "drift.csv", index=False)
        tg = pd.DataFrame(targ)
        tg.to_csv(out / "targets.csv", index=False)
        truth = a.truth or tg.member.iloc[0]
        po = ser[(ser.member == truth) & (ser.scenario == "ssp245")].set_index("year").loc[1850:2023, "tas"]
        po.rename("GMST").to_csv(out / "pseudo_obs.csv")
        print(f"== {model} (truth {truth})")
        print(pd.DataFrame(drift)[["member", "branch_year", "branch_ok", "parent_variant_label",
                                   "tas_slope_K_per_century", "N_offset", "N_slope_per_century"]]
              .round(3).to_string(index=False))
        print(tg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

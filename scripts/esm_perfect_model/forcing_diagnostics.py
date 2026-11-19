"""Forcing diagnostics for the perfect-model test.

Stage `components`:
    Run the first --n-members posterior members of one ESM's calibration through the four SSPs
    (concentration-driven, as in `project`) with every ERF component saved. Compare the median
    against RCMIP v5.1.0 ERF (what the RCMIP protocol prescribes as the reference forcing).
    -> output/esm_perfect_model_summary/forcing_components.csv, forcing_components.png

Stage `esm-forced`:
    Drive each ESM's 500 posterior members with that ESM's own diagnosed effective forcing
    F = N - lambda*T (ensemble mean, 1850-2100, anomaly vs piControl), for lambda from abrupt-4xCO2
    years 1-150 and years 1-20. Compare warming 2081-2100 vs 1995-2014 with the ESM and with the
    standard (concentration-driven) projection.
    -> output/esm_perfect_model_summary/esm_forced_scores.csv, esm_forced.png

Usage:
  PM_VARIANT=baseline PM_WORKERS=40 .venv-scm/bin/python scripts/esm_perfect_model/forcing_diagnostics.py components
  PM_VARIANT=baseline PM_WORKERS=40 .venv-scm/bin/python scripts/esm_perfect_model/forcing_diagnostics.py esm-forced
Both take --config (default histonly_fixedskew).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import run_perfect_model as R  # noqa: E402  (sets up the pinned ciceroscm)

sys.argv = _argv
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ciceroscm.parallel.distributionrun import DistributionRun  # noqa: E402

import summary_figures as SF  # noqa: E402

ROOT = R.ROOT
OUT = ROOT / "output" / "esm_perfect_model_summary"
RCMIP_ERF = Path("/storage/no-backup-nac/users/bensan/rcmip3_protocol/input_datafiles_generation/data/"
                 "rcmip-radiative-forcing-annual-means-v5-1-0.csv")
ERF = "Effective Radiative Forcing"
A = ERF + "|Anthropogenic"
# label: (SCM variables, RCMIP variables)
COMPONENTS = {
    "CO2": ([A + "|CO2"], [A + "|CO2"]),
    "CH4": ([A + "|CH4"], [A + "|CH4"]),
    "N2O": ([A + "|N2O"], [A + "|N2O"]),
    "Tropospheric O3": ([A + "|Ozone|Tropospheric Contribution"], [A + "|Tropospheric Ozone"]),
    "Stratospheric O3": ([A + "|Ozone|Stratospheric Contribution"], [A + "|Stratospheric Ozone"]),
    "Strat. H2O": ([A + "|Other|Stratospheric H2O"], [A + "|Other|CH4 Oxidation Stratospheric H2O"]),
    "Aerosol": ([A + "|Aerosol"], [A + "|Aerosols"]),
    "Halogenated + other": (None, None),  # residual of anthropogenic total
    "Natural": (None, None),  # total minus anthropogenic
    "Anthropogenic total": ([A], [A]),
    "Total": ([ERF], [ERF]),
}
SCM_VARS = sorted({v for s, _ in COMPONENTS.values() if s for v in s} | {"Atmospheric Concentrations|CO2"})
FUT, REF = (2081, 2100), (1995, 2014)
SAT = R.SAT


def draws(model, config, n=None):
    d = json.load(open(R.RUNS / model / "output" / f"draw_samples_{R.N_DRAW}_{config}.json"))
    return d[:n] if n else d


def write_draws(d, path):
    json.dump(d, open(path, "w"))
    return str(path)


def by_component(df):
    """df indexed by variable (SCM or RCMIP names, mapped via `which`), columns years -> component frame."""
    out = {}
    for lab, (sv, _) in COMPONENTS.items():
        if sv:
            out[lab] = df.loc[sv].sum()
    named = [c for c in out if c not in ("Anthropogenic total", "Total")]
    out["Halogenated + other"] = out["Anthropogenic total"] - sum(out[c] for c in named)
    out["Natural"] = out["Total"] - out["Anthropogenic total"]
    return pd.DataFrame(out).T.loc[list(COMPONENTS)]


def wmean(s, w):
    return s.loc[w[0]:w[1]].mean()


def stage_components(model, config, n):
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    f = write_draws(draws(model, config, n), work / f"draws_{model}_{n}.json")
    rc = pd.read_csv(RCMIP_ERF)
    rc = rc[rc.Region == "World"]
    rows, ts = [], {}
    for scen in R.SSPS:
        res = DistributionRun(None, json_file_name=f, numvalues=n).run_over_distribution(
            R.scendata(scen, 2100), SCM_VARS, max_workers=R.WORKERS)
        yrs = [c for c in res.columns if str(c).isdigit()]
        med = res.groupby("variable")[yrs].median()
        med.columns = med.columns.astype(int)
        scm = by_component(med)
        r = rc[rc.Scenario == scen].set_index("Variable")
        ry = [c for c in r.columns if str(c).isdigit() and 1750 <= int(c) <= 2100]
        r = r[ry].astype(float)
        r.columns = r.columns.astype(int)
        # RCMIP names, mapped onto our labels
        ref = {}
        for lab, (_, rv) in COMPONENTS.items():
            if rv:
                ref[lab] = r.loc[rv].sum()
        named = [c for c in ref if c not in ("Anthropogenic total", "Total")]
        ref["Halogenated + other"] = ref["Anthropogenic total"] - sum(ref[c] for c in named)
        ref["Natural"] = ref["Total"] - ref["Anthropogenic total"]
        ref = pd.DataFrame(ref).T.loc[list(COMPONENTS)]
        ts[scen] = (scm, ref)
        for lab in COMPONENTS:
            rows.append({"scenario": scen, "component": lab,
                         "scm_2014": scm.loc[lab, 2014], "rcmip_2014": ref.loc[lab, 2014],
                         "scm_2100": scm.loc[lab, 2100], "rcmip_2100": ref.loc[lab, 2100],
                         "scm_dF": wmean(scm.loc[lab], FUT) - wmean(scm.loc[lab], REF),
                         "rcmip_dF": wmean(ref.loc[lab], FUT) - wmean(ref.loc[lab], REF)})
        co2 = res[res.variable == "Atmospheric Concentrations|CO2"][yrs].median()
        print(f"{scen}: SCM CO2 conc 2014 {co2['2014'] if '2014' in co2 else co2[2014]:.1f}, "
              f"2100 {co2['2100'] if '2100' in co2 else co2[2100]:.1f} ppm", flush=True)
    t = pd.DataFrame(rows)
    t["diff_dF"] = t.scm_dF - t.rcmip_dF
    t.to_csv(OUT / "forcing_components.csv", index=False)
    pd.set_option("display.width", 200)
    for scen in R.SSPS:
        print(f"\n{scen} (W m-2; dF = 2081-2100 minus 1995-2014)")
        print(t[t.scenario == scen].drop(columns="scenario").round(2).to_string(index=False))
    fig_components(ts, model, config, n)


def fig_components(ts, model, config, n):
    labs = [c for c in COMPONENTS if c not in ("Anthropogenic total",)]
    fig, axs = plt.subplots(len(R.SSPS), len(labs), figsize=(2.1 * len(labs), 2.0 * len(R.SSPS)),
                            sharex=True)
    for i, scen in enumerate(R.SSPS):
        scm, ref = ts[scen]
        for j, lab in enumerate(labs):
            ax = axs[i, j]
            ax.plot(ref.columns, ref.loc[lab], color="0.3", lw=1.2, label="RCMIP v5.1.0")
            ax.plot(scm.columns, scm.loc[lab], color="#D55E00", lw=1.2, label="SCM posterior median")
            ax.set_xlim(1850, 2100)
            ax.grid(color="0.9", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{SF.SSP_LABEL[scen]}\nW m$^{{-2}}$", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"ERF components: SCM ({R.VARIANT}, {config}, first {n} posterior members for {model}) "
                 "vs RCMIP v5.1.0", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "forcing_components.png", dpi=130)
    plt.close(fig)


def esm_forcing(model, fb):
    es = pd.read_csv(R.TARGETS / model / "series.csv")
    out = {}
    for key, lam in [("l150", fb.loc[model, "lambda"]), ("l20", fb.loc[model, "lambda_1_20"])]:
        es["F"] = es.N - lam * es.tas
        for scen in R.SSPS:
            e = es[es.scenario == scen].groupby("year")[["F", "tas"]].mean()
            if e.empty:
                continue
            e = e.loc[1850:2100]
            if e.index[0] != 1850 or e.index[-1] != 2100 or len(e) != 251:
                continue
            out[(key, scen)] = e
    return out


def stage_esm_forced(models, config):
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    fb = pd.read_csv(OUT / "esm_ecs_lambda.csv", index_col=0)
    sc = pd.read_csv(OUT / "scores_all.csv")
    sc = sc[(sc.variant == R.VARIANT) & (sc.cfg == config) & (sc.window == SF.WINDOW)]
    f_out = OUT / f"esm_forced_scores_{R.VARIANT}_{config}.csv"
    done = pd.read_csv(f_out) if f_out.exists() else pd.DataFrame(columns=["model"])
    rows = done.to_dict("records")
    for m in models:
        if m in set(done.model) or m not in fb.index:
            continue
        dpath = R.RUNS / m / "output" / f"draw_samples_{R.N_DRAW}_{config}.json"
        if not dpath.exists():
            continue
        forc = esm_forcing(m, fb)
        gaspam = R.scendata("ssp245", 2100)[0]["gaspam_data"]  # only needed by the result reader
        for (key, scen), e in forc.items():
            sd = {"forc_data": e.F.to_numpy(), "nystart": 1850, "nyend": 2100, "emstart": 2100,
                  "sunvolc": 0, "scenname": f"{scen}_{key}", "idtm": 24,
                  "gaspam_data": gaspam}
            res = DistributionRun(None, json_file_name=str(dpath), numvalues=R.N_DRAW).run_over_distribution(
                sd, [SAT], max_workers=R.WORKERS)
            yrs = [c for c in res.columns if str(c).isdigit()]
            x = res[yrs].astype(float)
            x.columns = x.columns.astype(int)
            hist = x.loc[:, 2004:2023].mean(axis=1) - x.loc[:, 1850:1900].mean(axis=1)
            d = x.loc[:, FUT[0]:FUT[1]].mean(axis=1) - x.loc[:, REF[0]:REF[1]].mean(axis=1)
            esm_d = wmean(e.tas, FUT) - wmean(e.tas, REF)
            esm_h = wmean(e.tas, (2004, 2023)) - wmean(e.tas, (1850, 1900))
            q = d.quantile([0.05, 0.17, 0.5, 0.83, 0.95])
            s = sc[(sc.model == m) & (sc.scenario == scen)]
            rows.append({"model": m, "lambda": key, "scenario": scen,
                         "scm_p5": q[0.05], "scm_p17": q[0.17], "scm_p50": q[0.5], "scm_p83": q[0.83],
                         "scm_p95": q[0.95], "esm_ens_mean": esm_d, "pit": (d < esm_d).mean(),
                         "hist_scm_p50": hist.median(), "hist_esm": esm_h,
                         "conc_p50": s.scm_p50.iloc[0] if len(s) else np.nan,
                         "conc_p5": s.scm_p5.iloc[0] if len(s) else np.nan,
                         "conc_p95": s.scm_p95.iloc[0] if len(s) else np.nan})
            print(f"{m} {scen} {key}: ESM {esm_d:.2f}, SCM(ESM forcing) {q[0.05]:.2f}/{q[0.5]:.2f}/"
                  f"{q[0.95]:.2f}, SCM(conc) {rows[-1]['conc_p50']:.2f}; hist SCM {hist.median():.2f} "
                  f"vs ESM {esm_h:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(f_out, index=False)
    t = pd.DataFrame(rows)
    summarise_esm_forced(t, config)


def summarise_esm_forced(t, config):
    t = t.dropna(subset=["scm_p50", "esm_ens_mean"]).copy()
    t["in595"] = (t.esm_ens_mean >= t.scm_p5) & (t.esm_ens_mean <= t.scm_p95)
    t["bias"] = t.scm_p50 - t.esm_ens_mean
    t["conc_bias"] = t.conc_p50 - t.esm_ens_mean
    t["conc_in595"] = (t.esm_ens_mean >= t.conc_p5) & (t.esm_ens_mean <= t.conc_p95)
    t["hist_bias"] = t.hist_scm_p50 - t.hist_esm
    g = t.groupby(["lambda", "scenario"]).agg(n=("model", "size"), in595=("in595", "mean"),
                                              conc_in595=("conc_in595", "mean"), pit=("pit", "median"),
                                              bias=("bias", "median"), conc_bias=("conc_bias", "median"),
                                              hist_bias=("hist_bias", "median"))
    print(g.round(2).to_string())
    g.to_csv(OUT / f"esm_forced_summary_{R.VARIANT}_{config}.csv")
    # figure: bias with ESM forcing vs bias with SCM forcing
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    cols = dict(zip(R.SSPS, ["#0072B2", "#009E73", "#E69F00", "#D55E00"]))
    for ax, key, title in zip(axs, ["l150", "l20"], ["λ from years 1–150", "λ from years 1–20"]):
        tt = t[t["lambda"] == key]
        for scen in R.SSPS:
            d = tt[tt.scenario == scen]
            ax.scatter(d.conc_bias, d.bias, color=cols[scen], s=25, label=SF.SSP_LABEL[scen])
        lim = [-1.5, 2.5]
        ax.plot(lim, lim, color="0.6", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Bias, SCM with its own forcing (K)")
        ax.set_title(f"ESM forcing N − λT, {title}", fontsize=10)
        ax.grid(color="0.9", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Bias, SCM driven by ESM forcing (K)")
    axs[0].legend(frameon=False)
    fig.suptitle(f"Warming 2081–2100 vs 1995–2014, SCM median − ESM ensemble mean ({R.VARIANT}, {config})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"esm_forced_{R.VARIANT}_{config}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["components", "esm-forced"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--config", default="histonly_fixedskew")
    ap.add_argument("--n-members", type=int, default=50)
    a = ap.parse_args()
    if a.stage == "components":
        stage_components((a.models or ["MPI-ESM1-2-LR"])[0], a.config, a.n_members)
    else:
        fb = pd.read_csv(OUT / "esm_ecs_lambda.csv", index_col=0)
        stage_esm_forced(a.models or sorted(fb.index), a.config)

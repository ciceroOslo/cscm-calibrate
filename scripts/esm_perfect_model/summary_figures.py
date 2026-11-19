"""Cross-model summary of the perfect-model test, plus an ESM-vs-SCM forcing check.

Outputs (output/esm_perfect_model_summary/):
  esm_ecs_lambda.csv          ESM ECS and feedback (Gregory, abrupt-4xCO2 years 1-150 vs piControl mean)
  scores_all.csv              all score*.csv rows, with variant / config / model columns
  projections_<variant>.png   SCM 5-95/17-83/median for 2081-2100 vs 1995-2014 in each ESM, four
                              configurations side by side, ESMs sorted by ECS
  forcing_check.csv           ESM effective forcing (N + lambda T) vs SCM ERF, per model and scenario
  forcing_timeseries.png      ESM effective forcing vs SCM ERF median/5-95, per model
  forcing_vs_bias.png         forcing mismatch vs temperature bias across ESMs

ESM effective forcing uses the 4xCO2 Gregory feedback, which underestimates the historical
feedback in most models (pattern effect), so it is a rough check, not an exact forcing.

Usage: .venv-scm/bin/python scripts/esm_perfect_model/summary_figures.py [--forcing-variant baseline]
       [--forcing-config histonly_fixedskew]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GM = ROOT / "data" / "esm_perfect_model" / "global_means"
TARGETS = ROOT / "data" / "esm_perfect_model" / "targets"
OUT = ROOT / "output" / "esm_perfect_model_summary"
VARIANT_DIRS = {"ar7": "esm_perfect_model", "baseline": "esm_perfect_model_baseline",
                "efficacy": "esm_perfect_model_efficacy"}
VARIANT_LABEL = {"ar7": "ar7 (efficacy + aerosol feedback)", "baseline": "baseline (no efficacy, no aerosol feedback)",
                 "efficacy": "+ε (efficacy, no aerosol feedback)"}
CONFIGS = ["ecs", "fixedskew", "histonly", "histonly_fixedskew"]
CONFIG_LABEL = {"ecs": "ECS/TCR + history, pipeline weighting",
                "fixedskew": "ECS/TCR + history, skew fixed",
                "histonly": "history only, pipeline weighting",
                "histonly_fixedskew": "history only, skew fixed"}
# Okabe-Ito, colourblind-safe
CONFIG_COLOR = {"ecs": "#56B4E9", "fixedskew": "#0072B2", "histonly": "#E69F00", "histonly_fixedskew": "#D55E00"}
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
SSP_LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
WINDOW = "2081-2100 vs 1995-2014"
FUT, REF = (2081, 2100), (1995, 2014)
ERF = "Effective Radiative Forcing"


def esm_feedbacks():
    rows = {}
    for f in sorted(GM.glob("*.csv")):
        if f.stem.endswith("_meta"):
            continue
        d = pd.read_csv(f)
        if not {"piControl", "abrupt-4xCO2"} <= set(d.experiment_id):
            continue
        p = d[d.experiment_id.isin(["piControl", "abrupt-4xCO2"])].pivot_table(
            index=["experiment_id", "year"], columns="variable", values="value")
        p["N"] = p.rsdt - p.rsut - p.rlut
        c, x = p.loc["piControl"], p.loc["abrupt-4xCO2"].iloc[:150]
        ok = x[["tas", "N"]].dropna()
        dT, dN = ok.tas - c.tas.mean(), ok.N - c.N.mean()
        slope, icpt = np.polyfit(dT, dN, 1)
        slope20, icpt20 = np.polyfit(dT.iloc[:20], dN.iloc[:20], 1)
        rows[f.stem] = {"ECS": -icpt / slope / 2, "lambda": slope, "F4x": icpt,
                        "lambda_1_20": slope20, "F4x_1_20": icpt20}
    return pd.DataFrame(rows).T


def load_scores():
    rows = []
    for v, d in VARIANT_DIRS.items():
        for f in (ROOT / "output" / d).glob("*/score*.csv"):
            model = f.parent.name
            if model.endswith("prune1M"):
                continue
            s = pd.read_csv(f)
            s["variant"], s["model"] = v, model
            s["cfg"] = f.stem[len("score"):].lstrip("_") or "ecs"
            rows.append(s)
    return pd.concat(rows, ignore_index=True)


def fig_projections(sc, fb, variant, models):
    s = sc[(sc.variant == variant) & (sc.window == WINDOW)]
    x = np.arange(len(models))
    off = dict(zip(CONFIGS, np.linspace(-0.3, 0.3, len(CONFIGS))))
    fig, axs = plt.subplots(len(SSPS), 1, figsize=(15, 13), sharex=True)
    for ax, scen in zip(axs, SSPS):
        ss = s[s.scenario == scen].set_index(["model", "cfg"])
        for i, m in enumerate(models):
            ax.axvspan(i - 0.45, i + 0.45, color="0.95", lw=0, zorder=0) if i % 2 == 0 else None
            for c in CONFIGS:
                xi = i + off[c]
                if (m, c) not in ss.index:
                    ax.plot(xi, 0.02, marker="x", ms=4, color=CONFIG_COLOR[c], alpha=0.6,
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    continue
                r = ss.loc[(m, c)]
                ax.vlines(xi, r.scm_p5, r.scm_p95, color=CONFIG_COLOR[c], lw=1.2, alpha=0.6)
                ax.vlines(xi, r.scm_p17, r.scm_p83, color=CONFIG_COLOR[c], lw=4)
                ax.hlines(r.scm_p50, xi - 0.06, xi + 0.06, color="white", lw=1.5, zorder=3)
            r = s[(s.scenario == scen) & (s.model == m)]
            if len(r):
                r = r.iloc[0]
                ax.vlines(i, r.esm_min, r.esm_max, color="k", lw=0.8, zorder=4)
                ax.plot(i, r.truth, "o", color="k", ms=5, zorder=5, mec="white", mew=0.8)
        ax.set_ylabel(f"{SSP_LABEL[scen]}\nΔSAT (K)")
        ax.grid(axis="y", color="0.85", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    labels = [f"{m}\n{fb.ECS[m]:.1f}" if m in fb.index else f"{m}\n(no 4xCO2)" for m in models]
    axs[-1].set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    axs[-1].set_xlim(-0.6, len(models) - 0.4)
    axs[-1].set_xlabel("ESM (sorted by ECS from abrupt-4xCO2, K)")
    handles = [plt.Line2D([], [], color=CONFIG_COLOR[c], lw=4, label=CONFIG_LABEL[c]) for c in CONFIGS]
    handles += [plt.Line2D([], [], marker="o", color="k", ls="", label="ESM truth member"),
                plt.Line2D([], [], color="k", lw=0.8, label="ESM member range"),
                plt.Line2D([], [], marker="x", color="0.4", ls="", label="no result (pruning/ESS failure)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"Perfect-model test, {VARIANT_LABEL[variant]}: SCM 5–95% (thin), 17–83% (thick), "
                 f"median (white) for warming {WINDOW}", y=0.955, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"projections_{variant}.png", dpi=150)
    plt.close(fig)


def annual_window(y, w):
    return y.loc[w[0]:w[1]].mean()


def forcing_check(fb, variant, cfg, models):
    """ESM F = N + |lambda| T (ensemble mean) vs SCM ERF, both relative to 1850-1900."""
    rows, series = [], {}
    pdir_name = "projections" + ("" if cfg == "ecs" else f"_{cfg}")
    for m in models:
        pdir = ROOT / "output" / VARIANT_DIRS[variant] / m / pdir_name
        if m not in fb.index or not pdir.exists() or not (TARGETS / m / "series.csv").exists():
            continue
        es = pd.read_csv(TARGETS / m / "series.csv")
        es["F"] = es.N - fb.loc[m, "lambda"] * es.tas
        es["F20"] = es.N - fb.loc[m, "lambda_1_20"] * es.tas
        for scen in SSPS:
            f = pdir / f"{scen}.csv"
            e = es[es.scenario == scen].groupby("year")[["F", "F20", "tas"]].mean()
            if e.empty or not f.exists():
                continue
            e = e - e.loc[1850:1900].mean()
            p = pd.read_csv(f)
            p = p[p.variable == ERF]
            yrs = [c for c in p.columns if c.isdigit()]
            erf = p[yrs].astype(float)
            erf.columns = erf.columns.astype(int)
            erf = erf.sub(erf.loc[:, 1850:1900].mean(axis=1), axis=0)
            q = erf.quantile([0.05, 0.5, 0.95])
            series[(m, scen)] = (e, q)
            fe = annual_window(e.F, FUT) - annual_window(e.F, REF)
            fs = annual_window(q.loc[0.5], FUT) - annual_window(q.loc[0.5], REF)
            fh_e = annual_window(e.F, REF) - 0  # 1995-2014 vs 1850-1900
            fh_s = annual_window(q.loc[0.5], REF)
            fe20 = annual_window(e.F20, FUT) - annual_window(e.F20, REF)
            rows.append({"model": m, "scenario": scen, "dF_esm": fe, "dF_esm_l20": fe20, "dF_scm": fs,
                         "Fhist_esm": fh_e, "Fhist_scm": fh_s})
    return pd.DataFrame(rows), series


def fig_forcing_timeseries(series, fb, models):
    ms = [m for m in models if (m, "ssp245") in series]
    ncol = 5
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(17, 3.0 * nrow), sharex=True, sharey=True)
    cols = {"ssp126": "#0072B2", "ssp585": "#D55E00"}
    for ax, m in zip(axs.flat, ms):
        for scen, c in cols.items():
            if (m, scen) not in series:
                continue
            e, q = series[(m, scen)]
            ax.fill_between(q.columns, q.loc[0.05], q.loc[0.95], color=c, alpha=0.15, lw=0)
            ax.plot(q.columns, q.loc[0.5], color=c, lw=1.5)
            ax.plot(e.index, e.F.rolling(11, center=True, min_periods=6).mean(), color=c, lw=1.2, ls="--")
        ax.set_title(f"{m} (ECS {fb.ECS[m]:.1f})", fontsize=9)
        ax.set_xlim(1850, 2100)
        ax.axvline(2015, color="0.6", lw=0.5)
        ax.grid(color="0.9", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs.flat[len(ms):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("W m$^{-2}$ vs 1850–1900")
    handles = [plt.Line2D([], [], color=cols["ssp126"], lw=1.5, label="SSP1-2.6"),
               plt.Line2D([], [], color=cols["ssp585"], lw=1.5, label="SSP5-8.5"),
               plt.Line2D([], [], color="0.3", lw=1.5, label="SCM ERF median (shading 5–95%)"),
               plt.Line2D([], [], color="0.3", lw=1.2, ls="--",
                          label="ESM N − λT, ensemble mean, 11-yr running mean")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "forcing_timeseries.png", dpi=130)
    plt.close(fig)


def fig_forcing_vs_bias(fc, sc, variant, cfg):
    s = sc[(sc.variant == variant) & (sc.cfg == cfg) & (sc.window == WINDOW)]
    d = fc.dropna(subset=["dF_esm"]).merge(s[["model", "scenario", "scm_p50", "esm_mean"]], on=["model", "scenario"])
    d["dT_bias"] = d.scm_p50 - d.esm_mean
    d["dF_bias"] = d.dF_scm - d.dF_esm
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cols = dict(zip(SSPS, ["#0072B2", "#009E73", "#E69F00", "#D55E00"]))
    ax = axs[0]
    for scen in SSPS:
        dd = d[d.scenario == scen]
        ax.hlines(dd.dF_scm, dd.dF_esm, dd.dF_esm_l20, color=cols[scen], lw=0.8, alpha=0.6)
        ax.scatter(dd.dF_esm, dd.dF_scm, color=cols[scen], s=25, label=SSP_LABEL[scen])
    lim = [min(d.dF_esm.min(), d.dF_scm.min()) - 0.3, max(d.dF_esm.max(), d.dF_scm.max()) + 0.3]
    ax.plot(lim, lim, color="0.5", lw=0.8)
    ax.set_xlabel("ESM effective forcing change, N − λT (dot: λ yrs 1–150; line: to λ yrs 1–20), W m$^{-2}$")
    ax.set_ylabel("SCM ERF change (posterior median), W m$^{-2}$")
    ax.set_title(f"Forcing change {WINDOW}", fontsize=10)
    ax.legend(frameon=False)
    ax = axs[1]
    for scen in SSPS:
        dd = d[d.scenario == scen]
        ax.scatter(dd.dF_bias, dd.dT_bias, color=cols[scen], s=25)
    for _, r in d[d.scenario == "ssp245"].iterrows():
        ax.annotate(r.model, (r.dF_bias, r.dT_bias), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ok = d[["dF_bias", "dT_bias"]].dropna()
    rr = np.corrcoef(ok.dF_bias, ok.dT_bias)[0, 1]
    ax.set_xlabel("Forcing change, SCM − ESM (W m$^{-2}$)")
    ax.set_ylabel("Warming, SCM median − ESM mean (K)")
    ax.set_title(f"Forcing mismatch vs warming bias (r = {rr:.2f}; labels SSP2-4.5)", fontsize=10)
    for a in axs:
        a.grid(color="0.9", lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{VARIANT_LABEL[variant]}, {CONFIG_LABEL[cfg]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "forcing_vs_bias.png", dpi=150)
    plt.close(fig)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing-variant", default="baseline")
    ap.add_argument("--forcing-config", default="histonly_fixedskew")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fb = esm_feedbacks()
    fb.to_csv(OUT / "esm_ecs_lambda.csv")
    sc = load_scores()
    sc.to_csv(OUT / "scores_all.csv", index=False)
    models = sorted(sc.model.unique(), key=lambda m: (fb.ECS.get(m, 99), m))
    # include models with no results at all (pruning failures)
    allm = sorted({p.parent.name for p in (ROOT / "output" / VARIANT_DIRS["ar7"]).glob("*/prune_meta.json")}
                  - {"MPI-ESM1-2-LR_prune1M"})
    models = sorted(set(models) | set(allm), key=lambda m: (fb.ECS.get(m, 99), m))
    for v in VARIANT_DIRS:
        fig_projections(sc, fb, v, models)
    fc, series = forcing_check(fb, a.forcing_variant, a.forcing_config, models)
    fig_forcing_timeseries(series, fb, models)
    d = fig_forcing_vs_bias(fc, sc, a.forcing_variant, a.forcing_config)
    d.to_csv(OUT / "forcing_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby("scenario")[["dF_esm", "dF_scm", "dF_bias", "dT_bias"]].median().round(2))
    print(d.groupby("scenario")[["dF_esm_l20"]].median().round(2))
    print(d[d.scenario == "ssp245"][["model", "Fhist_esm", "Fhist_scm", "dF_esm", "dF_esm_l20", "dF_scm", "dT_bias"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()

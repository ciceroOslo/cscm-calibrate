"""Figures for the CMIP7 per-ESM joint fits (v2: discrepancy model, structural variants).

    python scripts/cmip7_esm/plot_fits.py [--variants eps eps_state] [sources ...]

fit_<variant>_<source>.png  full fit: GSAT and N per fitted tier, infill of all seven tiers,
                            abrupt-4xCO2 (T, Gregory) and 1pctCO2
holdout_summary.png         each held-out tier, per variant: ESM 2081-2100 warming vs the
                            emulator's predictive range (thin: incl. discrepancy; thick:
                            parameters only) from the fit without that tier
holdout_timeseries.png      held-out tier GSAT, ESM vs emulator from the fit without it
posteriors.png              derived ECS, TCR, F4x, aerosol ERF, efficacy, state dependence
holdout_summary.csv         the numbers behind holdout_summary.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emulator as EM  # noqa: E402
import esm_targets as TG  # noqa: E402
import scm_forcing as SF  # noqa: E402

FITS = SF.OUT / "fits"
FIG = SF.OUT / "figures"
TIER_COLORS = {"VL": "#16188F", "LN": "#22e5db", "L": "#20A359", "ML": "#dec820",
               "M": "#fc7b03", "HL": "#E744F6", "H": "#a41212"}
TIER_LABELS = {"VL": "Very low", "LN": "Low–net-zero", "L": "Low", "ML": "Medium-low",
               "M": "Medium", "HL": "High–low", "H": "High"}
VARIANT_LABELS = {"eps": "+ε", "eps_state": "+ε, state-dependent feedback"}
VARIANT_COLORS = {"eps": "#0072B2", "eps_state": "#D55E00"}  # Okabe-Ito
INK, MUTED, GRID = "0.15", "0.45", "0.88"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
                     "legend.frameon": False, "savefig.dpi": 200})


def _dir(source, variant, tag, driver="concentration"):
    return FITS / f"{source}_{driver[:4]}_{variant}_{tag}"


def _done(d):
    return (d / "scores.csv").exists()


def _pred(d):
    return pd.read_csv(d / "predictions.csv.gz")


def _smooth(v, n):
    return pd.Series(v).rolling(n, center=True, min_periods=1).mean().to_numpy() if n > 1 else np.asarray(v)


def _band(ax, p, color, label=None, smooth=1, total=True):
    x = p.x.to_numpy()
    if total and "p05_tot" in p:
        ax.fill_between(x, _smooth(p.p05_tot, smooth), _smooth(p.p95_tot, smooth), color=color, alpha=0.10, lw=0)
    ax.fill_between(x, _smooth(p.p05, smooth), _smooth(p.p95, smooth), color=color, alpha=0.25, lw=0)
    ax.plot(x, _smooth(p.p50, smooth), color=color, lw=1.0, ls="--", label=label)


def fig_fit(source, variant, driver="concentration"):
    d = _dir(source, variant, "full", driver)
    if not _done(d):
        return
    pr = _pred(d)
    meta = json.load(open(d / "meta.json"))
    blocks = TG.load(source, driver)
    fitted = meta["tiers"]
    sel = lambda b, v: pr[(pr.block == b) & (pr["var"] == v)]  # noqa: E731
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    a = ax[0, 0]
    for t in fitted:
        p = sel(t, "T")
        _band(a, p[p.x >= 1850], TIER_COLORS[t])
        b = blocks[f"{t}/T"]
        a.plot(b["x"], b["y"], color=TIER_COLORS[t], lw=0.9, label=f"{TIER_LABELS[t]} (k={b['k']})")
    h = blocks["hist/T"]
    a.plot(h["x"], h["y"], color="0.4", lw=0.9, label="historical")
    a.set(title="GSAT: ESM mean (solid), emulator (dashed; dark 5–95% params, light + discrepancy)",
          ylabel="K vs 1850–1900")
    a.legend(loc="upper left", fontsize=7)
    a = ax[0, 1]
    for t in SF.TIERS:
        p = sel(t, "T")
        _band(a, p[p.x >= 1990], TIER_COLORS[t], label=TIER_LABELS[t] + ("" if t in fitted else " (infilled)"),
              total=False)
    a.set(title="Emulated GSAT, all seven tiers (5–95%, parameters only)", ylabel="K vs 1850–1900")
    a.legend(loc="upper left", fontsize=6.5)
    a = ax[0, 2]
    for t in fitted:
        _band(a, sel(t, "N").query("x >= 1850"), TIER_COLORS[t], smooth=11, total=False)
        if f"{t}/N" in blocks:
            b = blocks[f"{t}/N"]
            a.plot(b["x"], _smooth(b["y"], 11), color=TIER_COLORS[t], lw=0.9)
    if "hist/N" in blocks:
        h = blocks["hist/N"]
        a.plot(h["x"], _smooth(h["y"], 11), color="0.4", lw=0.9)
    a.set(title="Net TOA imbalance N (11-yr running mean)", ylabel="W m$^{-2}$")
    a = ax[1, 0]
    _band(a, sel("abrupt-4xCO2", "T").query("x <= 300"), "0.2")
    b = blocks.get("abrupt-4xCO2/T")
    if b:
        a.plot(b["x"], b["y"], color="0.2", lw=0.8)
    a.set(title="abrupt-4xCO2 GSAT", xlabel="year", ylabel="K")
    a = ax[1, 1]
    bT, bN = blocks.get("abrupt-4xCO2/T"), blocks.get("abrupt-4xCO2/N")
    pT, pN = sel("abrupt-4xCO2", "T").set_index("x"), sel("abrupt-4xCO2", "N").set_index("x")
    n = len(bT["x"]) if bT else 150
    a.plot(pT.p50.iloc[:n], pN.p50.iloc[:n], color="0.2", ls="--", lw=1.0, label="emulator median")
    if bT and bN:
        a.scatter(bT["y"], bN["y"], s=4, color="0.35", lw=0, label="ESM")
    m = TG.metrics(source)
    dg = pd.read_csv(d / "diagnostics.csv")
    q = lambda c: f"{dg[c].median():.2f} [{dg[c].quantile(.05):.2f}–{dg[c].quantile(.95):.2f}]"  # noqa: E731
    a.set(title=f"abrupt-4x Gregory: ESM ECS {m.get('ECS', np.nan):.2f} K; emulator {q('ECS_gregory')}",
          xlabel="T (K)", ylabel="N (W m$^{-2}$)")
    a.legend(fontsize=7)
    a = ax[1, 2]
    _band(a, sel("1pctCO2", "T").query("x <= 140"), "0.2")
    b = blocks.get("1pctCO2/T")
    if b:
        a.plot(b["x"], b["y"], color="0.2", lw=0.8)
    a.set(title=f"1pctCO2: ESM TCR {m.get('TCR', np.nan):.2f} K; emulator {q('TCR')}", xlabel="year", ylabel="K")
    for a in ax.flat:
        a.title.set_fontsize(7)
    fig.suptitle(f"{source} ({driver}-driven): CICERO-SCM {VARIANT_LABELS[variant]} fitted jointly to history, "
                 f"{', '.join(fitted)} and idealised runs", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"fit_{variant}_{source}.png")
    plt.close(fig)


def holdout_table(sources, variants, driver="concentration"):
    rows = []
    for v in variants:
        for s in sources:
            full = _dir(s, v, "full", driver)
            if not _done(full):
                continue
            fs = pd.read_csv(full / "scores.csv").set_index("block")
            for t in json.load(open(full / "meta.json"))["tiers"]:
                d = _dir(s, v, "no" + t, driver)
                if not _done(d):
                    continue
                hs = pd.read_csv(d / "scores.csv").set_index("block")
                k = f"{t}/T"
                cols = ("emu_p50", "par_p05", "par_p95", "tot_p05", "tot_p95", "pit_tot", "rmse", "frac_in_90")
                rows.append({"variant": v, "source": s, "tier": t, "esm": hs.loc[k, "esm_2081_2100"],
                             "held_err": hs.loc[k, "emu_p50"] - hs.loc[k, "esm_2081_2100"],
                             "full_err": fs.loc[k, "emu_p50"] - fs.loc[k, "esm_2081_2100"],
                             **{f"held_{c}": hs.loc[k, c] for c in cols},
                             **{f"full_{c}": fs.loc[k, c] for c in ("tot_p05", "tot_p95", "pit_tot")}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(SF.OUT / "holdout_summary.csv", index=False)
    return df


def fig_holdout(df, variants, driver="concentration"):
    cases = df[["source", "tier"]].drop_duplicates().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    offs = np.linspace(-0.15, 0.15, len(variants))
    for i, c in cases.iterrows():
        for off, v in zip(offs, variants):
            r = df[(df.variant == v) & (df.source == c.source) & (df.tier == c.tier)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = VARIANT_COLORS[v]
            ax.plot([i + off] * 2, [r.held_tot_p05, r.held_tot_p95], color=col, lw=1)
            ax.plot([i + off] * 2, [r.held_par_p05, r.held_par_p95], color=col, lw=3.5, solid_capstyle="butt")
            ax.plot(i + off, r.held_emu_p50, "o", color=col, mfc="white", ms=4,
                    label=VARIANT_LABELS[v] if i == 0 else None)
        esm = df[(df.source == c.source) & (df.tier == c.tier)].esm.iloc[0]
        ax.plot([i - 0.3, i + 0.3], [esm] * 2, color=INK, lw=1.5, label="ESM ensemble mean" if i == 0 else None)
    ax.set_xticks(range(len(cases)), [f"{s.split('-')[0]}\n{TIER_LABELS[t]}" for s, t in zip(cases.source, cases.tier)])
    ax.set(ylabel="GSAT 2081–2100 (K vs 1850–1900)",
           title="Held-out tier: emulator 5–95% (thick: parameters; thin: + discrepancy) vs ESM")
    ax.title.set_fontsize(8)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "holdout_summary.png")
    plt.close(fig)
    fig, axs = plt.subplots(len(variants), len(cases), figsize=(2.3 * len(cases), 2.3 * len(variants)),
                            constrained_layout=True, squeeze=False)
    for j, v in enumerate(variants):
        for i, c in cases.iterrows():
            a = axs[j, i]
            d = _dir(c.source, v, "no" + c.tier, driver)
            if not _done(d):
                a.set_visible(False)
                continue
            p = _pred(d)
            p = p[(p.block == c.tier) & (p["var"] == "T") & (p.x >= 1980)]
            _band(a, p, TIER_COLORS[c.tier])
            b = TG.load(c.source, driver)[f"{c.tier}/T"]
            a.plot(b["x"], b["y"], color=INK, lw=0.8)
            a.set_title(f"{c.source.split('-')[0]} {TIER_LABELS[c.tier]} held out\n{VARIANT_LABELS[v]}", fontsize=7)
        axs[j, 0].set_ylabel("GSAT (K)")
    fig.savefig(FIG / "holdout_timeseries.png")
    plt.close(fig)


def fig_posteriors(sources, variants, driver="concentration"):
    keys = [("ECS_gregory", "ECS (Gregory 1–150, K)", "ECS"), ("TCR", "TCR (K)", "TCR"),
            ("F4x", "F$_{4×}$ (W m$^{-2}$)", None), ("ERF_aer_2019", "aerosol ERF 2019 (W m$^{-2}$)", None)]
    pars = [("ocean_efficacy", "ocean heat-uptake efficacy ε"), ("a_state", "a (W m$^{-2}$ K$^{-2}$)")]
    fits = [_dir(s, v, t, driver) for v in variants for s in sources for t in ("full", "noH", "noM", "noVL")]
    fits = [d for d in fits if _done(d)]
    if not fits:
        return
    fig, axs = plt.subplots(1, len(keys) + len(pars), figsize=(13, 0.32 * len(fits) + 1.3),
                            constrained_layout=True, sharey=True)
    for j, d in enumerate(fits):
        meta = json.load(open(d / "meta.json"))
        dg = pd.read_csv(d / "diagnostics.csv")
        post = np.load(d / "posterior.npy")
        m = TG.metrics(meta["source"])
        col = VARIANT_COLORS[meta["variant"]]
        vals = [dg[k] for k, _, _ in keys] + [post[:, EM.NAMES.index(p)] for p, _ in pars]
        marks = [m.get(mk) if mk else None for _, _, mk in keys] + [None] * len(pars)
        labs = [lab for _, lab, _ in keys] + [lab for _, lab in pars]
        for a, val, mk, lab in zip(axs, vals, marks, labs):
            q = np.percentile(val, [5, 50, 95])
            a.plot([q[0], q[2]], [j, j], color=col, lw=1.5)
            a.plot(q[1], j, "o", color=col, ms=3)
            if mk is not None:
                a.plot(mk, j, "|", color=INK, ms=9, mew=2)
            a.set_xlabel(lab)
    labels = [f"{json.load(open(d / 'meta.json'))['source'].split('-')[0]} {d.name.rsplit('_', 1)[1]}"
              f" ({VARIANT_LABELS[json.load(open(d / 'meta.json'))['variant']]})" for d in fits]
    axs[0].set_yticks(range(len(fits)), labels)
    axs[0].invert_yaxis()
    axs[0].set_title("posterior 5–50–95%; black tick = ESM value", fontsize=7, loc="left")
    fig.savefig(FIG / "posteriors.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", default=["CanESM6-0-MR", "UKESM1-3-LL"])
    ap.add_argument("--variants", nargs="+", default=["eps", "eps_state"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        for s in a.sources:
            fig_fit(s, v)
    df = holdout_table(a.sources, a.variants)
    if not df.empty:
        fig_holdout(df, a.variants)
        cols = ["variant", "source", "tier", "esm", "full_err", "held_err", "held_par_p05", "held_par_p95",
                "held_tot_p05", "held_tot_p95", "held_pit_tot"]
        print(df[cols].round(2).to_string(index=False))
    fig_posteriors(a.sources, a.variants)

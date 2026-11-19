"""Rank (PIT) summary: where each ESM member's warming falls in the SCM posterior distribution.

PIT = fraction of the 500 SCM posterior members warming less than the ESM member, for warming in
--window (default 2081-2100 vs 1995-2014). Every ESM member counts, weighted 1/n_members so each
ESM has equal weight. A calibrated ensemble gives a flat histogram / diagonal CDF; PIT piled near 0
means the SCM is too warm, U-shaped means too narrow.

Outputs (output/esm_perfect_model_summary/):
  pit_<window>.csv                one row per variant/config/scenario/model/member
  rank_hist_<variant>_<window>.png  histograms, rows = configurations, columns = scenarios
  rank_cdf_<window>.png           CDFs of PIT, rows = variants, columns = scenarios, one line per
                                  configuration

Usage: .venv-scm/bin/python scripts/esm_perfect_model/rank_summary.py [--window 2041-2060]
       .venv-scm/bin/python scripts/esm_perfect_model/rank_summary.py --esm-forced
         (rank_cdf_esm_forced.png: same members driven by each ESM's own diagnosed forcing)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import summary_figures as SF  # noqa: E402

ROOT = SF.ROOT
OUT = SF.OUT
SAT = "Surface Air Temperature Change"
REF = (1995, 2014)
NBIN = 10


def wmean(df, w):
    cols = [c for c in df.columns if w[0] <= c <= w[1]]
    return df[cols].mean(axis=1)


def compute(fut):
    rows = []
    for v, d in SF.VARIANT_DIRS.items():
        for mdir in sorted((ROOT / "output" / d).iterdir()):
            m = mdir.name
            ser = SF.TARGETS / m / "series.csv"
            if m.endswith("prune1M") or not ser.exists():
                continue
            es = pd.read_csv(ser)
            for cfg in SF.CONFIGS:
                pdir = mdir / ("projections" + ("" if cfg == "ecs" else f"_{cfg}"))
                if not pdir.exists():
                    continue
                for scen in SF.SSPS:
                    f = pdir / f"{scen}.csv"
                    if not f.exists():
                        continue
                    p = pd.read_csv(f)
                    p = p[p.variable == SAT]
                    yrs = [c for c in p.columns if c.isdigit()]
                    x = p[yrs].astype(float)
                    x.columns = x.columns.astype(int)
                    scm = np.sort((wmean(x, fut) - wmean(x, REF)).to_numpy())
                    e = es[es.scenario == scen].pivot_table(index="member", columns="year", values="tas")
                    e = e.dropna(axis=0, subset=[y for y in e.columns if REF[0] <= y <= fut[1]])
                    if e.empty:
                        continue
                    de = wmean(e, fut) - wmean(e, REF)
                    for mem, val in de.items():
                        pit = np.searchsorted(scm, val) / len(scm)
                        rows.append({"variant": v, "cfg": cfg, "scenario": scen, "model": m,
                                     "member": mem, "esm": val, "pit": pit, "w": 1 / len(de)})
    return pd.DataFrame(rows)


def fig_hist(t, variant, tag):
    fig, axs = plt.subplots(len(SF.CONFIGS), len(SF.SSPS), figsize=(13, 10), sharex=True, sharey=True)
    bins = np.linspace(0, 1, NBIN + 1)
    for i, cfg in enumerate(SF.CONFIGS):
        for j, scen in enumerate(SF.SSPS):
            ax = axs[i, j]
            d = t[(t.variant == variant) & (t.cfg == cfg) & (t.scenario == scen)]
            nm = d.model.nunique()
            if nm:
                h, _ = np.histogram(d.pit, bins=bins, weights=d.w)
                ax.bar(bins[:-1], h / nm * NBIN, width=1 / NBIN, align="edge",
                       color=SF.CONFIG_COLOR[cfg], edgecolor="white", lw=1)
            ax.axhline(1, color="0.3", lw=0.8, ls="--")
            ax.text(0.97, 0.93, f"{nm} ESMs", transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 1)
            if i == 0:
                ax.set_title(SF.SSP_LABEL[scen], fontsize=10)
            if j == 0:
                ax.set_ylabel(SF.CONFIG_LABEL[cfg].replace(", ", ",\n"), fontsize=8)
            if i == len(SF.CONFIGS) - 1:
                ax.set_xlabel("PIT of ESM member in SCM posterior")
    fig.suptitle(f"Rank histograms, {SF.VARIANT_LABEL[variant]}, warming {tag}\n"
                 "(density, each ESM weighted equally; dashed = calibrated; mass near 0 = SCM too warm)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"rank_hist_{variant}_{tag.split()[0]}.png", dpi=140)
    plt.close(fig)


def fig_cdf(t, tag):
    variants = list(SF.VARIANT_DIRS)
    fig, axs = plt.subplots(len(variants), len(SF.SSPS), figsize=(13, 9.5), sharex=True, sharey=True)
    grid = np.linspace(0, 1, 201)
    for i, v in enumerate(variants):
        for j, scen in enumerate(SF.SSPS):
            ax = axs[i, j]
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
            ax.axvspan(0.05, 0.95, color="0.95", lw=0, zorder=0)
            for cfg in SF.CONFIGS:
                d = t[(t.variant == v) & (t.cfg == cfg) & (t.scenario == scen)]
                if d.empty:
                    continue
                cdf = [(d.w * (d.pit <= g)).sum() / d.w.sum() for g in grid]
                ax.plot(grid, cdf, color=SF.CONFIG_COLOR[cfg], lw=1.8,
                        label=f"{SF.CONFIG_LABEL[cfg]} (n={d.model.nunique()})")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="0.9", lw=0.5)
            if i == 0:
                ax.set_title(SF.SSP_LABEL[scen], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{v}\nCumulative fraction of ESMs", fontsize=9)
            if i == len(variants) - 1:
                ax.set_xlabel("PIT of ESM member in SCM posterior")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, [x.split(" (n=")[0] for x in lab], loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"PIT CDFs, warming {tag}: diagonal = calibrated, above diagonal = SCM too warm; "
                 "shading = 5–95% range", y=0.955, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"rank_cdf_{tag.split()[0]}.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="2081-2100")
    ap.add_argument("--esm-forced", action="store_true", help="only the ESM-forced comparison")
    a = ap.parse_args()
    if a.esm_forced:
        forced_main()
        return
    fut = tuple(int(x) for x in a.window.split("-"))
    tag = f"{a.window} vs {REF[0]}-{REF[1]}"
    t = compute(fut)
    t.to_csv(OUT / f"pit_{a.window}.csv", index=False)
    for v in SF.VARIANT_DIRS:
        fig_hist(t, v, tag)
    fig_cdf(t, tag)

    def stats(d):
        w = d.w / d.w.sum()
        return pd.Series({"n_esm": d.model.nunique(), "mean_pit": (w * d.pit).sum(),
                          "frac_<0.05": (w * (d.pit < 0.05)).sum(), "frac_>0.95": (w * (d.pit > 0.95)).sum(),
                          "frac_in_5_95": (w * d.pit.between(0.05, 0.95)).sum()})

    s = t.groupby(["variant", "cfg", "scenario"]).apply(stats)
    s.to_csv(OUT / f"pit_summary_{a.window}.csv")
    pd.set_option("display.width", 200)
    print(s.round(2).to_string())



# ---------------------------------------------------------------- SCM driven by ESM forcing
FORCED = [("baseline", "histonly_fixedskew"), ("efficacy", "histonly_fixedskew"), ("ar7", "histonly_fixedskew"),
          ("baseline", "fixedskew"), ("ar7", "fixedskew")]
FORCING_STYLE = {"conc": ("SCM's own (RCMIP) forcing", "0.2", "-"),
                 "l150": ("ESM forcing N − λT, λ yrs 1–150", "#0072B2", "-"),
                 "l20": ("ESM forcing N − λT, λ yrs 1–20", "#D55E00", "-")}


def forced_table():
    """PIT of the ESM ensemble mean (2081-2100 vs 1995-2014) under each forcing."""
    sc = pd.read_csv(OUT / "scores_all.csv")
    sc = sc[sc.window == SF.WINDOW]
    rows = []
    for v, c in FORCED:
        f = OUT / f"esm_forced_scores_{v}_{c}.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f).dropna(subset=["scm_p50"])
        models = set(e.model)
        for _, r in e.iterrows():
            rows.append({"variant": v, "cfg": c, "scenario": r.scenario, "model": r.model, "forcing": r["lambda"],
                         "pit": r.pit, "bias": r.scm_p50 - r.esm_ens_mean,
                         "in595": r.scm_p5 <= r.esm_ens_mean <= r.scm_p95})
        s = sc[(sc.variant == v) & (sc.cfg == c) & sc.model.isin(models)]
        for _, r in s.iterrows():
            rows.append({"variant": v, "cfg": c, "scenario": r.scenario, "model": r.model, "forcing": "conc",
                         "pit": r.esm_mean_pit, "bias": r.scm_p50 - r.esm_mean,
                         "in595": bool(r.esm_mean_in_5_95)})
    return pd.DataFrame(rows)


def fig_forced(t):
    combos = [vc for vc in FORCED if ((t.variant == vc[0]) & (t.cfg == vc[1])).any()]
    fig, axs = plt.subplots(len(combos), len(SF.SSPS), figsize=(13, 2.6 * len(combos)), sharex=True, sharey=True)
    grid = np.linspace(0, 1, 201)
    for i, (v, c) in enumerate(combos):
        for j, scen in enumerate(SF.SSPS):
            ax = axs[i, j]
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
            ax.axvspan(0.05, 0.95, color="0.95", lw=0, zorder=0)
            for k, (lab, col, ls) in FORCING_STYLE.items():
                d = t[(t.variant == v) & (t.cfg == c) & (t.scenario == scen) & (t.forcing == k)]
                if d.empty:
                    continue
                ax.plot(grid, [(d.pit <= g).mean() for g in grid], color=col, ls=ls, lw=1.8, label=lab)
                ax.text(0.97, 0.05 + 0.1 * list(FORCING_STYLE).index(k), f"{d.in595.mean():.0%} in 5–95",
                        color=col, transform=ax.transAxes, ha="right", fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="0.9", lw=0.5)
            if i == 0:
                ax.set_title(SF.SSP_LABEL[scen], fontsize=10)
            if j == 0:
                n = t[(t.variant == v) & (t.cfg == c)].model.nunique()
                ax.set_ylabel(f"{v}, {SF.CONFIG_LABEL[c].split(', ')[0]}\n(n={n})\ncum. fraction", fontsize=8)
            if i == len(combos) - 1:
                ax.set_xlabel("PIT of ESM ensemble mean")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Same calibrated SCM members, driven by their own forcing vs each ESM's diagnosed forcing "
                 "(warming 2081–2100 vs 1995–2014, skew fixed)", y=0.965, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "rank_cdf_esm_forced.png", dpi=140)
    plt.close(fig)


def forced_main():
    t = forced_table()
    t.to_csv(OUT / "pit_esm_forced.csv", index=False)
    fig_forced(t)
    s = t.groupby(["variant", "cfg", "forcing", "scenario"]).agg(
        n=("model", "size"), mean_pit=("pit", "mean"), in595=("in595", "mean"), bias=("bias", "median"))
    s.to_csv(OUT / "pit_esm_forced_summary.csv")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()

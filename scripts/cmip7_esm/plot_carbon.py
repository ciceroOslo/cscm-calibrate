"""Figures for the stage-2 (emissions-driven) carbon-cycle fits.

carbon_fits.png      per ESM (rows): CO2, ocean uptake, GSAT; ESM ensemble mean (solid) vs
                     emulator full fit (dashed median, dark 5-95% params, light + discrepancy)
carbon_holdout.png   2081-2100 CO2 and GSAT for each held-out emissions-driven tier
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esm_targets as TG  # noqa: E402
import fit_carbon as FC  # noqa: E402
import plot_fits as PF  # noqa: E402

SOURCES = ["CanESM6-0-MR", "UKESM1-3-LL"]
VARS = [("C", "CO$_2$ (ppm)"), ("O", "ocean uptake (PgC yr$^{-1}$)"), ("T", "GSAT (K vs 1850–1900)")]


def fig_fits(variant="eps"):
    fig, axs = plt.subplots(len(SOURCES), 3, figsize=(10, 3 * len(SOURCES)), constrained_layout=True)
    for r, src in enumerate(SOURCES):
        d = FC.run_dir(src, variant, None)
        pr = pd.read_csv(d / "predictions.csv.gz")
        bl = TG.load_emissions(src)
        for a, (v, lab) in zip(axs[r], VARS):
            for t in FC.E_TIERS:
                p = pr[(pr.block == t) & (pr["var"] == v) & (pr.x >= 1850)]
                sm = 11 if v == "O" else 1
                PF._band(a, p, PF.TIER_COLORS[t], smooth=sm)
                if f"{t}/{v}" in bl:
                    b = bl[f"{t}/{v}"]
                    a.plot(b["x"], PF._smooth(b["y"], sm), color=PF.TIER_COLORS[t], lw=0.9, label=PF.TIER_LABELS[t])
            h = bl.get(f"hist/{v}")
            if h:
                a.plot(h["x"], PF._smooth(h["y"], 11 if v == "O" else 1), color="0.4", lw=0.9, label="esm-hist")
            a.set(ylabel=lab, title=f"{src}: {lab.split(' (')[0]}")
            a.title.set_fontsize(8)
        axs[r, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Emissions-driven: CICERO-SCM +ε (stage-1 thermal) with carbon cycle fitted to esm-hist, H, M, VL "
                 "(solid ESM, dashed emulator)", fontsize=9)
    fig.savefig(PF.FIG / "carbon_fits.png")
    plt.close(fig)


def fig_holdout(variant="eps"):
    rows = []
    for src in SOURCES:
        for t in FC.E_TIERS:
            d = FC.run_dir(src, variant, t)
            if not (d / "scores.csv").exists():
                continue
            s = pd.read_csv(d / "scores.csv").set_index("block")
            f = pd.read_csv(FC.run_dir(src, variant, None) / "scores.csv").set_index("block")
            for v in ("C", "T"):
                k = f"{t}/{v}"
                rows.append({"source": src, "tier": t, "var": v, "esm": s.loc[k, "esm_2081_2100"],
                             "held_p50": s.loc[k, "emu_p50"], "held_p05": s.loc[k, "tot_p05"],
                             "held_p95": s.loc[k, "tot_p95"], "held_pit": s.loc[k, "pit_tot"],
                             "full_p50": f.loc[k, "emu_p50"], "full_pit": f.loc[k, "pit_tot"]})
    df = pd.DataFrame(rows)
    df.to_csv(FC.OUT / "carbon_holdout_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for a, (v, lab) in zip(axs, [("C", "CO$_2$ 2081–2100 (ppm)"), ("T", "GSAT 2081–2100 (K)")]):
        x = df[df["var"] == v].reset_index(drop=True)
        for i, r in x.iterrows():
            c = PF.TIER_COLORS[r.tier]
            a.plot([i, i], [r.held_p05, r.held_p95], color=c, lw=2)
            a.plot(i, r.held_p50, "o", color=c, mfc="white", ms=5, label="emulator, tier held out" if i == 0 else None)
            a.plot(i + 0.2, r.full_p50, "s", color=c, ms=3, label="emulator, full fit (median)" if i == 0 else None)
            a.plot([i - 0.3, i + 0.3], [r.esm] * 2, color=PF.INK, lw=1.5, label="ESM" if i == 0 else None)
        a.set_xticks(range(len(x)), [f"{s.split('-')[0]}\n{PF.TIER_LABELS[t]}" for s, t in zip(x.source, x.tier)],
                     fontsize=6.5)
        a.set(ylabel=lab)
    axs[0].legend(fontsize=6.5, loc="upper left")
    axs[0].set_title("Held-out emissions-driven tier: 5–95% incl. discrepancy", fontsize=8, loc="left")
    fig.savefig(PF.FIG / "carbon_holdout.png")
    plt.close(fig)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    fig_fits()
    fig_holdout()

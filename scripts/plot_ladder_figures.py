#!/usr/bin/env python
"""Generate the cumulative-ladder comparison figures for the paper.

Self-contained companion to ``extract_ladder_metrics.py`` (it reuses that
module's loaders and metric functions). Reads Marit's dumps read-only by
absolute path and writes PNGs into ``analysis/figures/`` in the paper repo, so
the manuscript can ``\\includegraphics`` them in place of the placeholder
floats. Can be copied verbatim into ``cscm-calibrate/scripts`` once the
remaining (+PDO) dumps exist; just extend the ``LADDER`` registry in
``extract_ladder_metrics.py``.

Produces:
  ladder_sensitivity_marginals.png  -> fig:posterior_marginals (ECS, TCR)
  ladder_projection_violins.png     -> fig:projection_spread
  ladder_flat10_metrics.png         -> fig:flat10_metrics (TCRE, ZEC50)

Usage:
    python plot_ladder_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import extract_ladder_metrics as elm

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Consistent colour per rung across all figures.
COLOURS = {
    "0 baseline (v1)": "#444444",
    "1 +efficacy": "#1f77b4",
    "2 +pattern": "#d62728",
    "4a IPCC+pattern": "#2ca02c",
}


def per_member(rung_dump: str):
    """Return dict of per-member metric arrays for one rung's dump."""
    dump = elm.DUMP_ROOT / rung_dump
    out = {}
    a4 = elm.load_scenario(dump, "abrupt-4xCO2")
    if a4 is not None:
        out["ECS"] = elm.ecs_gregory(a4).dropna().values
    p1 = elm.load_scenario(dump, "1pctCO2")
    if p1 is not None:
        out["TCR"] = elm.transient(p1, 69).dropna().values
    for tag, scen in elm.PROJ_SCENARIOS.items():
        df = elm.load_scenario(dump, scen)
        if df is not None:
            out[tag] = elm.eoc_warming(df).dropna().values
    return out


def main() -> None:
    data = {label: per_member(dump) for label, dump, _, _ in elm.LADDER}
    rungs = list(data.keys())

    # ── Figure 1: ECS and TCR marginals ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for var, ax in zip(("ECS", "TCR"), axes):
        for r in rungs:
            v = data[r].get(var)
            if v is None:
                continue
            ax.hist(v, bins=40, range=(0, 10 if var == "ECS" else 5),
                    histtype="step", density=True, lw=2,
                    color=COLOURS.get(r), label=r)
            ax.axvline(np.median(v), color=COLOURS.get(r), ls=":", lw=1)
        ax.set_xlabel(f"{var} (K)")
        ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Climate sensitivity along the structural ladder")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ladder_sensitivity_marginals.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: end-of-century projection distributions ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (tag, title), ax in zip(
        [("scen7H", "scen7-H (high)"), ("scen7VL", "scen7-VL (very low)")], axes
    ):
        series, labels, colours = [], [], []
        for r in rungs:
            v = data[r].get(tag)
            if v is not None:
                series.append(v)
                labels.append(r)
                colours.append(COLOURS.get(r))
        bp = ax.boxplot(series, whis=(5, 95), showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        for box, c in zip(bp["boxes"], colours):
            box.set_facecolor(c)
            box.set_alpha(0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(r"$\Delta T_{2081-2100}$ vs 1850-1900 (K)")
        ax.set_title(title)
    fig.suptitle("End-of-century warming along the structural ladder "
                 "(box: IQR; whiskers: 5-95%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ladder_projection_violins.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: flat10 carbon-cycle metrics (TCRE, ZEC50) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, xlabel in (
        ("tcre", axes[0], r"TCRE (K EgC$^{-1}$)"),
        ("zec50", axes[1], r"ZEC$_{50}$ (K)"),
    ):
        for label, _, suffix, _ in elm.LADDER:
            fm = elm.flat10_metrics(suffix)
            if fm is None or col not in fm:
                continue
            ax.hist(fm[col].dropna(), bins=40, histtype="step", density=True,
                    lw=2, color=COLOURS.get(label), label=label)
            ax.axvline(fm[col].median(), color=COLOURS.get(label), ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(r"\texttt{flat10MIP} carbon-cycle metrics along the ladder"
                 .replace(r"\texttt{", "").replace("}", ""))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ladder_flat10_metrics.png", dpi=150)
    plt.close(fig)

    print(f"wrote 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

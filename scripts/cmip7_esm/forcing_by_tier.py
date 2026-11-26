"""ESM-diagnosed effective forcing per tier vs the fitted emulator's forcing.

ESM: F = N - lambda*T with lambda from the ESM's own abrupt-4xCO2 Gregory fit (years
1-150 and 1-20 to bracket the pattern effect). Emulator: the stage-1 forcing
(scaled SCM components, global mean, excl. volcanic) at posterior draws.

    python scripts/cmip7_esm/forcing_by_tier.py [--variant eps]
writes output/cmip7_esm/forcing_by_tier.csv and figures/forcing_by_tier.png
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
import fit_esm as F1  # noqa: E402
import plot_fits as PF  # noqa: E402
import scm_forcing as SF  # noqa: E402


def gregory(source, n):
    g = pd.read_csv(TG.DATA / "gregory.csv")
    g = g[(g.source == source) & (g.year_index <= n)].dropna(subset=["dT_K", "dN_Wm2"])
    return np.polyfit(g.dT_K, g.dN_Wm2, 1)[0]


def emu_forcing(comps, tier, theta):
    s = dict(zip(EM.NAMES, theta))
    g = lambda k: comps[f"{tier}/{k}"]  # noqa: E731
    hemi = lambda k: 0.5 * (g(f"{k}_n") + g(f"{k}_s"))  # noqa: E731
    return s["s_co2"] * g("co2") + s["s_ari"] * hemi("ari") + s["s_aci"] * hemi("aci") + s["s_other"] * hemi("other") + g("solar")


def main(variant, sources):
    comps = SF.load()
    years = np.arange(SF.NYSTART, SF.NYEND + 1)
    rows = []
    fig, axs = plt.subplots(1, len(sources), figsize=(5 * len(sources), 3.4), constrained_layout=True, squeeze=False)
    for a, src in zip(axs[0], sources):
        d = F1.run_dir(src, "concentration", variant, None)
        post = np.load(d / "posterior.npy")
        draws = post[np.random.default_rng(0).choice(len(post), 200, replace=False)]
        lam = {"150": gregory(src, 150), "20": gregory(src, 20)}
        blocks = TG.load(src)
        for tier in [t for t in json.load(open(d / "meta.json"))["tiers"]]:
            T, N = blocks[f"{tier}/T"], blocks[f"{tier}/N"]
            x = T["x"]
            Fe = {k: N["y"] - v * T["y"] for k, v in lam.items()}
            Fm = np.stack([emu_forcing(comps, tier, th) for th in draws])[:, np.searchsorted(years, x)]
            w = x >= 2081
            r = {"source": src, "tier": tier, "esm_F_l150": Fe["150"][w].mean(), "esm_F_l20": Fe["20"][w].mean(),
                 "emu_F_p50": np.median(Fm[:, w].mean(axis=1)), "emu_F_p05": np.percentile(Fm[:, w].mean(axis=1), 5),
                 "emu_F_p95": np.percentile(Fm[:, w].mean(axis=1), 95), "esm_T": T["y"][w].mean()}
            rows.append(r)
            c = PF.TIER_COLORS[tier]
            a.plot(x, PF._smooth(Fe["150"], 11), color=c, lw=1, label=f"{PF.TIER_LABELS[tier]} ESM (λ$_{{150}}$)")
            a.plot(x, PF._smooth(Fe["20"], 11), color=c, lw=0.8, ls=":")
            a.fill_between(x, np.percentile(Fm, 5, axis=0), np.percentile(Fm, 95, axis=0), color=c, alpha=0.2, lw=0)
            a.plot(x, np.median(Fm, axis=0), color=c, ls="--", lw=1)
        a.set(title=f"{src}: ESM effective forcing (solid λ150, dotted λ20) vs emulator (dashed, 5–95%)",
              ylabel="W m$^{-2}$")
        a.title.set_fontsize(7)
        a.legend(fontsize=6.5, loc="upper left")
    fig.savefig(PF.FIG / "forcing_by_tier.png")
    df = pd.DataFrame(rows)
    df.to_csv(SF.OUT / "forcing_by_tier.csv", index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="eps")
    ap.add_argument("sources", nargs="*", default=["CanESM6-0-MR", "UKESM1-3-LL"])
    a = ap.parse_args()
    main(a.variant, a.sources)

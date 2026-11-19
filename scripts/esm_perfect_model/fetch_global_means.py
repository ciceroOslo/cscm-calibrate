"""Download CMIP6 Amon tas/rsdt/rsut/rlut with intake-esgf, reduce to annual global means, and
keep only those. Raw files go to a cache on /scratch and are deleted once a model is done
(unless --keep-raw).

Member choice (per model, from inventory_models / inventory_datasets.csv):
  historical + SSPs   up to --members members that are complete (all 4 vars) in historical,
                      ranked by the number of SSPs they also have, then r/i/p/f order
  piControl,          1 member (the first complete one; parent_variant_label of the
  abrupt-4xCO2        historical runs is stored in the metadata so this can be checked)
  esm-hist +          only if listed in --experiments; up to --members members of esm-hist,
  esm-ssp585          ranked by whether they also have esm-ssp585
  esm-piControl       1 member (searched live; not in the inventory)

Outputs (data/esm_perfect_model/global_means/):
  <model>.csv        long format: source_id, experiment_id, member_id, grid_label, variable, year, value
                     (area-weighted with areacella, or cos(lat) if areacella is missing;
                      annual mean weighted by month length; incomplete years dropped)
  <model>_meta.csv   one row per dataset: branch times, parent info, calendar, weighting used

Raw global means only; drift correction against piControl is a separate step.

Usage:
  .venv/bin/python scripts/esm_perfect_model/fetch_global_means.py --models CanESM5 MIROC6
  .venv/bin/python scripts/esm_perfect_model/fetch_global_means.py --all-core --members 5
  .venv/bin/python scripts/esm_perfect_model/fetch_global_means.py --all-esm --members 5 \
      --experiments esm-hist esm-ssp585 esm-piControl
Reruns skip (model, experiment, member) combinations already in <model>.csv.
"""

import argparse
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import intake_esgf
from intake_esgf import ESGFCatalog

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "esm_perfect_model"
OUT = DATA / "global_means"
CACHE = Path("/scratch/bensan/esgf_cache")
VARS = ["tas", "rsdt", "rsut", "rlut"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
ESM_SCEN = ["esm-ssp585"]
ONE_MEMBER = ["piControl", "abrupt-4xCO2", "esm-piControl"]
LIVE_SEARCH = ["esm-piControl"]  # experiments not in the inventory files
DEFAULT_EXPS = ["historical"] + SSPS + ONE_MEMBER
# Always decode to cftime: piControl runs past 2262 otherwise mix Timestamp and cftime, and
# chunks="auto" fails on object-dtype time bounds.
OPEN_KW = {"chunks": {}, "decode_times": xr.coders.CFDatetimeCoder(use_cftime=True)}
META_ATTRS = ["parent_experiment_id", "parent_variant_label", "parent_time_units", "branch_time_in_parent",
              "branch_time_in_child", "branch_method", "parent_source_id"]


def member_key(m):
    return tuple(int(x) for x in re.findall(r"\d+", m))


def complete_members(inv, model, exp):
    d = inv[(inv.source_id == model) & (inv.experiment_id == exp)]
    if d.empty and exp in LIVE_SEARCH:  # not in the inventory: ask ESGF directly
        cat = ESGFCatalog()
        try:
            cat.search(project="CMIP6", source_id=model, experiment_id=exp, variable_id=VARS,
                       table_id="Amon", quiet=True)
            d = cat.df
        except Exception:  # intake-esgf raises when nothing is found
            return []
    n = d.groupby("member_id")["variable_id"].nunique()
    return sorted(n[n == len(VARS)].index, key=member_key)


def ranked_members(inv, model, base, followers, n_members):
    """Up to n_members complete members of `base`, preferring those also in `followers`."""
    sets = [set(complete_members(inv, model, e)) for e in followers]
    ranked = sorted(complete_members(inv, model, base),
                    key=lambda m: (-sum(m in s for s in sets), member_key(m)))
    return sorted(ranked[:n_members], key=member_key)


def choose_members(inv, model, exps, n_members):
    plan = {}
    groups = {"historical": ("historical", SSPS), "esm-hist": ("esm-hist", ESM_SCEN)}
    chosen = {}
    for e in exps:
        avail = complete_members(inv, model, e)
        if not avail:
            continue
        if e in ONE_MEMBER:
            plan[e] = avail[:1]
            continue
        base = "esm-hist" if e.startswith("esm-") else "historical"
        if base not in chosen:
            chosen[base] = ranked_members(inv, model, base, groups[base][1], n_members)
        plan[e] = [m for m in chosen[base] if m in avail]
    if "esm-piControl" in plan and "esm-hist" not in plan:
        del plan["esm-piControl"]  # nothing to drift-correct
    return {e: m for e, m in plan.items() if m}


def annual_global_mean(ds, var):
    da = ds[var]
    spatial = [d for d in da.dims if d != "time"]
    if "areacella" in ds:
        w, wname = ds["areacella"].fillna(0), "areacella"
    else:
        lat = ds["lat"]
        w, wname = np.cos(np.deg2rad(lat)) * xr.ones_like(da.isel(time=0)), "coslat"
    gm = da.weighted(w).mean(spatial)
    dim = gm["time"].dt.days_in_month
    yr = gm["time"].dt.year
    gm = gm.compute()
    num = (gm * dim).groupby(yr).sum()
    den = dim.groupby(yr).sum()
    cnt = gm.groupby(yr).count()
    ann = (num / den).where(cnt == 12, drop=True)
    cal = getattr(gm["time"].values[0], "calendar", "standard")
    return pd.Series(ann.values, index=ann["year"].values.astype(int)), wname, cal


def done_set(model):
    f = OUT / f"{model}.csv"
    if not f.exists():
        return set(), f
    d = pd.read_csv(f, usecols=["experiment_id", "member_id", "variable"])
    n = d.drop_duplicates().groupby(["experiment_id", "member_id"])["variable"].nunique()
    return set(n[n == len(VARS)].index), f


def fetch_model(model, plan, keep_raw):
    done, f_out = done_set(model)
    f_meta = OUT / f"{model}_meta.csv"
    for exp, members in plan.items():
        todo = [m for m in members if (exp, m) not in done]
        if not todo:
            print(f"  {exp}: all {len(members)} member(s) already done", flush=True)
            continue
        t0 = time.time()
        try:
            cat = ESGFCatalog()
            cat.search(project="CMIP6", source_id=model, experiment_id=exp, member_id=todo,
                       variable_id=VARS, table_id="Amon", quiet=True)
            # one grid per variable/member: prefer gn
            df = cat.df
            if "grid_label" in df.columns and df["grid_label"].nunique() > 1:
                g = "gn" if "gn" in set(df.grid_label) else sorted(df.grid_label)[0]
                cat.df = df[df.grid_label == g].reset_index(drop=True)
            dsd = cat.to_dataset_dict(add_measures=True, minimal_keys=False, quiet=True,
                                      open_kwargs=OPEN_KW)
        except Exception:
            print(f"  {exp}: FAILED search/download\n{traceback.format_exc()}", flush=True)
            continue
        rows, meta = [], []
        for key, ds in dsd.items():
            parts = dict(zip(["project", "activity", "institution", "source_id", "experiment_id",
                              "member_id", "table_id", "variable", "grid_label"], key.split(".")))
            var = parts.get("variable")
            if var not in VARS:
                var = next((v for v in VARS if v in ds.data_vars), None)
            mem = parts.get("member_id", ds.attrs.get("variant_label"))
            try:
                s, wname, cal = annual_global_mean(ds, var)
            except Exception:
                print(f"  {key}: FAILED global mean\n{traceback.format_exc()}", flush=True)
                continue
            if s.empty:  # no complete year (e.g. truncated or partial download)
                print(f"  {key}: FAILED no complete years", flush=True)
                continue
            rows.append(pd.DataFrame({"source_id": model, "experiment_id": exp, "member_id": mem,
                                      "grid_label": parts.get("grid_label", ds.attrs.get("grid_label")),
                                      "variable": var, "year": s.index, "value": s.values}))
            meta.append({"source_id": model, "experiment_id": exp, "member_id": mem, "variable": var,
                         "weights": wname, "calendar": cal, "first_year": s.index.min(),
                         "last_year": s.index.max(), "dataset_key": key,
                         **{a: ds.attrs.get(a) for a in META_ATTRS}})
        if rows:
            pd.concat(rows).to_csv(f_out, mode="a", header=not f_out.exists(), index=False)
            pd.DataFrame(meta).to_csv(f_meta, mode="a", header=not f_meta.exists(), index=False)
        got = {(r.experiment_id.iloc[0], r.member_id.iloc[0], r.variable.iloc[0]) for r in rows}
        print(f"  {exp}: {len(got)} series for {len(todo)} member(s) "
              f"({len(got)}/{len(todo) * len(VARS)} expected), {time.time() - t0:.0f} s", flush=True)
    if not keep_raw:
        for p in CACHE.glob(f"CMIP6/*/*/{model}"):
            shutil.rmtree(p, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--all-core", action="store_true", help="all models with has_core+4x in inventory")
    ap.add_argument("--all-esm", action="store_true", help="all models with esm-ssp585 in inventory")
    ap.add_argument("--experiments", nargs="*", default=DEFAULT_EXPS)
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()

    inv = pd.read_csv(DATA / "inventory_datasets.csv")
    models = list(a.models)
    if a.all_core:
        tab = pd.read_csv(DATA / "inventory_models.csv", index_col=0)
        models += [m for m in tab.index[tab["has_core+4x"]] if m not in models]
    if a.all_esm:
        tab = pd.read_csv(DATA / "inventory_models.csv", index_col=0)
        models += [m for m in tab.index[tab["esm-ssp585"] > 0] if m not in models]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # The CMIP6 STAC indices have a known publishing error in member_id (intake-esgf warning),
    # so search the Solr indices instead.
    indices = {"discovery.east.esgf.io": False, "discovery.west.esgf.io": False,
               "esgf.ceda.ac.uk": True, "esgf-data.dkrz.de": True}
    with intake_esgf.conf.set(indices=indices, local_cache=[str(CACHE)], break_on_error=False,
                              num_threads=6):
        for i, model in enumerate(models, 1):
            plan = choose_members(inv, model, a.experiments, a.members)
            print(f"[{i}/{len(models)}] {model}: " + ", ".join(f"{e}×{len(m)}" for e, m in plan.items()),
                  flush=True)
            fetch_model(model, plan, a.keep_raw)


if __name__ == "__main__":
    main()

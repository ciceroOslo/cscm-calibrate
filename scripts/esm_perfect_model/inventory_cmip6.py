"""Inventory CMIP6 data available on ESGF for the SCM perfect-model (ESM-as-truth) test.

Search only, nothing is downloaded. Uses intake-esgf (the search/fetch layer the CMIP REF
uses) to find datasets, and the ESGF Solr index for dataset sizes.

Outputs (in data/esm_perfect_model/):
  inventory_datasets.csv  one row per (source_id, experiment_id, member_id, variable_id)
  inventory_models.csv    per model: members per experiment, and whether the full
                          concentration-driven set is present for all variables

Usage: .venv/bin/python scripts/esm_perfect_model/inventory_cmip6.py
"""

from pathlib import Path

import pandas as pd
import requests
from intake_esgf import ESGFCatalog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "esm_perfect_model"
OUT.mkdir(parents=True, exist_ok=True)

VARS = ["tas", "rsdt", "rsut", "rlut"]
CONC = ["historical", "ssp126", "ssp245", "ssp370", "ssp585", "piControl", "abrupt-4xCO2"]
ESM = ["esm-hist", "esm-ssp585", "esm-ssp534-over"]
CORE = ["historical", "ssp126", "ssp245", "ssp585", "piControl"]  # minimum for the conc test
SOLR = "https://esgf.ceda.ac.uk/esg-search/search"

# ---------------------------------------------------------------- intake-esgf search
cat = ESGFCatalog()
cat.search(project="CMIP6", experiment_id=CONC + ESM, variable_id=VARS, table_id="Amon", quiet=True)
df = cat.df.copy()
print("intake-esgf columns:", list(df.columns))
key = ["source_id", "experiment_id", "member_id", "variable_id"]
df = df[key + [c for c in ("grid_label", "version") if c in df.columns]].drop_duplicates(key)

# ---------------------------------------------------------------- sizes from ESGF Solr (latest; max over replicas)
rows = []
for exp in CONC + ESM:
    start = 0
    while True:
        r = requests.get(SOLR, params=dict(
            project="CMIP6", experiment_id=exp, variable_id=VARS, table_id="Amon", type="Dataset",
            latest="true", distrib="true", limit=5000, offset=start,
            fields="source_id,experiment_id,member_id,variable_id,size", format="application/solr+json",
        ), timeout=120).json()["response"]
        for d in r["docs"]:
            rows.append({k: (d[k][0] if isinstance(d.get(k), list) else d.get(k)) for k in key + ["size"]})
        start += len(r["docs"])
        if start >= r["numFound"] or not r["docs"]:
            break
sz = pd.DataFrame(rows).groupby(key, as_index=False)["size"].max()
df = df.merge(sz, on=key, how="left")
df["size_GB"] = df.pop("size") / 1e9
df.sort_values(key).to_csv(OUT / "inventory_datasets.csv", index=False)

# ---------------------------------------------------------------- per-model summary
complete = df.groupby(["source_id", "experiment_id", "member_id"])["variable_id"].nunique() == len(VARS)
mem = complete[complete].reset_index().groupby(["source_id", "experiment_id"])["member_id"].nunique()
tab = mem.unstack("experiment_id").reindex(columns=CONC + ESM).fillna(0).astype(int)
tab["has_core"] = (tab[CORE] > 0).all(axis=1)
tab["has_core+4x"] = tab["has_core"] & (tab["abrupt-4xCO2"] > 0)
# volume for the first complete member of each core experiment (+ abrupt-4xCO2), all variables
first = (complete[complete].reset_index().sort_values("member_id")
         .groupby(["source_id", "experiment_id"]).head(1)[["source_id", "experiment_id", "member_id"]])
vol = df.merge(first).query("experiment_id in @CORE or experiment_id == 'abrupt-4xCO2'")
tab["GB_1member"] = vol.groupby("source_id")["size_GB"].sum().reindex(tab.index).fillna(0).round(1)
tab["GB_1member_noPiC"] = (vol[vol.experiment_id != "piControl"].groupby("source_id")["size_GB"]
                           .sum().reindex(tab.index).fillna(0).round(1))
tab = tab.sort_values(["has_core+4x", "has_core"], ascending=False)
tab.to_csv(OUT / "inventory_models.csv")

pd.set_option("display.width", 250, "display.max_rows", 200)
print(tab.to_string())
for lab, m in [("core (hist+ssp126/245/585+piControl)", tab.has_core), ("core + abrupt-4xCO2", tab["has_core+4x"])]:
    t = tab[m]
    print(f"\n{lab}: {m.sum()} models; 1 member each: {t.GB_1member.sum():.0f} GB "
          f"({t.GB_1member_noPiC.sum():.0f} GB without piControl)")
print(f"esm-ssp585 (all 4 vars): {(tab['esm-ssp585'] > 0).sum()} models, {tab['esm-ssp585'].sum()} members")

"""Perfect-model (ESM-as-truth) calibration of CICERO-SCM with the cscm-calibrate pipeline.

Stages
  prior                 one concentration-driven prior ensemble, shared by all ESMs:
                        CMIP6 historical + ssp245 forcing (RCMIP inputs) 1750-2023, no pattern/PDO term,
                        prior parameter ranges from tests/test-data/config_file_ar7.json.
  idealised MODEL...    prune each model, then run abrupt-4xCO2 / 1pctCO2 once on the union of
                        their pruned members (ECS/TCR depend only on the parameter set, not on
                        the ESM). Results accumulate in <runs>/idealised/ecs_tcr.csv.
  harvest MODEL         add a model's own per-model idealised results to ecs_tcr.csv.
  calibrate MODEL       prune to the ESM pseudo-observed SAT (RMSE < 0.17 K, as AR7), take
                        ECS/TCR for the pruned members from ecs_tcr.csv (or run them for this
                        model if --own-idealised), weight to the pseudo-targets plus the assessed
                        ECS/TCR/aerosol ERF, draw 500.
                        --hist-only: weight to the historical targets (SAT, OHC, aerosol ERF)
                        only, no ECS/TCR constraint (paper's "+aerosol" calibration style).
  project MODEL         run the 500-member posterior on ssp126/245/370/585 to 2100.

PM_USE_CHUNKS limits pruning to the first k prior chunks (k x 10000 members); it is recorded
in <model>/prune_meta.json and a model directory pruned with a different k is refused.

Differences from the AR7 observational calibration (on purpose):
  * GMST target and pruning series use the SCM's Surface Air Temperature Change (ESM tas is SAT).
  * GMST and OHC targets are centred on the ESM's values (truth member) with the AR7 widths.
  * TCRE and CO2 concentration are not used (no carbon cycle in a concentration-driven test).
Everything else (prune threshold, ECS/TCR from idealised runs with blended temperature,
the weighting routine) is the pipeline's own code.

--fixed-skew uses a corrected skew-normal fit in the weighting (the pipeline passes
central -/+ sigma as the 5th/95th percentiles; sigma = (central - 5th)/1.645). Output then goes
to draw_samples_500_fixedskew.json and projections_fixedskew/.

Run directories: output/esm_perfect_model/{prior,<MODEL>}/output/ (pipeline layout; the prior
chunk files are symlinked into each model's directory).

Usage:
  .venv-scm/bin/python scripts/esm_perfect_model/run_perfect_model.py prior [--n 1000000]
  .venv-scm/bin/python scripts/esm_perfect_model/run_perfect_model.py calibrate MPI-ESM1-2-LR [--fixed-skew]
  .venv-scm/bin/python scripts/esm_perfect_model/run_perfect_model.py project MPI-ESM1-2-LR [--fixed-skew]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CICEROSCM = Path("/storage/no-backup-nac/users/bensan/cscm-calibrate/sst_hash_pattern_test/runA/ciceroscm")
INDATA = "/storage/no-backup-nac/users/bensan/cscm-calibrate/sst_hash_pattern_test/runA/indata/"
AR7_TARGETS = Path("/div/no-backup-nac/users/masan/GRAFITE/cscm-calibrate/data/"
                   "Calibration_targets_for_AR7_WG1_Ch5_emulators_Calibration_targets.csv")
# Structural variant = prior config (the variants of Sandstad et al., pattern-effect paper):
#   ar7       wide lambda + ocean efficacy + aerosol feedback   ("aerosol ECS constrained")
#   efficacy  wide lambda + ocean efficacy, delta_lambda_aero=0 ("+epsilon")
#   baseline  wide lambda, efficacy=1, delta_lambda_aero=0      ("baseline")
VARIANTS = {"ar7": "config_file_ar7.json",
            "efficacy": "config_file_no_deltaaero_wide_lambda.json",
            "baseline": "config_file_nopattern_no_efficacy.json"}
VARIANT = os.environ.get("PM_VARIANT", "ar7")
PRIOR_CONFIG = ROOT / "tests" / "test-data" / VARIANTS[VARIANT]
TARGETS = ROOT / "data" / "esm_perfect_model" / "targets"
_default_runs = ROOT / "output" / ("esm_perfect_model" + ("" if VARIANT == "ar7" else f"_{VARIANT}"))
RUNS = Path(os.environ.get("PM_RUNS", _default_runs))  # override for tests
GASES = "gases_vupdate_2024_WMO_added_new.txt"
SAT = "Surface Air Temperature Change"
CHUNK = 10000
WORKERS = int(os.environ.get("PM_WORKERS", 200))
N_DRAW = int(os.environ.get("PM_NDRAW", 500))  # override for tests
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PROJ_VARS = [SAT, "Surface Air Ocean Blended Temperature Change", "Heat Content|Ocean",
             "Effective Radiative Forcing", "Effective Radiative Forcing|Anthropogenic|Aerosol"]

# ------------------------------------------------------------------ pinned ciceroscm, then the pipeline
sys.path.insert(0, str(CICEROSCM / "src"))
import ciceroscm  # noqa: E402

assert str(CICEROSCM) in ciceroscm.__file__, ciceroscm.__file__
sys.path.insert(0, str(ROOT / "src"))
from cscm_calibrate import plot_distributions_w_obs as plots  # noqa: E402
from cscm_calibrate import prune_distribution_to_timeseries as prn  # noqa: E402
from cscm_calibrate import run_extra_experiments_post_pruning as rex  # noqa: E402
from cscm_calibrate import run_prior_ensemble as rpe  # noqa: E402
from cscm_calibrate import shared_functions as shf  # noqa: E402
from cscm_calibrate import weigth_ensemble_from_constraints_and_draw as wgt  # noqa: E402
from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm  # noqa: E402
from ciceroscm.parallel._configdistro import _ConfigDistro  # noqa: E402
from ciceroscm.parallel.distributionrun import DistributionRun  # noqa: E402

assert str(CICEROSCM) in sys.modules["ciceroscm.parallel.distributionrun"].__file__


def set_run_dir(d):
    """Point every pipeline module's output/ at d/output and work inside d."""
    d = Path(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    for mod in (shf, prn, rex, rpe, wgt, plots):
        mod.get_project_root = lambda d=d: str(d)
    os.chdir(d)
    return d / "output"


# ------------------------------------------------------------------ scenario data
def scendata(scen, nyend):
    sd = define_scendata_for_scm(
        INDATA, gaspam=GASES, df_conc=f"{scen}_conc_{GASES}", df_emis=f"{scen}_em_{GASES}",
        nystart=1750, nyend=nyend, emstart=nyend, sunvolc=1,
        rf_volc_file=f"VOLC_RCMIP_{scen}_RCMIP3.txt", rf_solar_file=f"solar_RCMIP_{scen}_RCMIP3.txt",
        rf_luc_file=f"LUCalbedo_RCMIP_{scen}_RCMIP3.txt",
    )
    sd[0]["conc_run"] = True
    sd[0]["scenname"] = scen
    return sd


def idealised_scendata(experiments, yearends):
    """As CSCMCalibrationPipeline.run_additional_experiments_post_pruning (conc-driven branch)."""
    out = []
    for exp, nyend in zip(experiments, yearends):
        sd = define_scendata_for_scm(
            INDATA, gaspam=GASES,
            df_nat_ch4=shf.make_dataframe_of_zeros("CH4", 1750, nyend + 1),
            df_nat_n2o=shf.make_dataframe_of_zeros("N2O", 1750, nyend + 1),
            df_conc=f"{exp}_conc_{GASES}", df_emis=f"esm-piControl_em_{GASES}",
            rf_luc_file="LUCalbedo_RCMIP_constant_zero_RCMIP3.txt",
            nystart=1750, emstart=nyend + 1, nyend=nyend, sunvolc=0,
        )[0]
        sd.update(conc_run=True, scenname=exp, ref_yr=int(min(nyend - 1, 2010)))
        out.append(sd)
    return out


# ------------------------------------------------------------------ constraints
def constraints(model=None):
    """AR7 targets with GMST (as SAT) and OHC re-centred on the ESM truth member; TCRE/CO2 dropped.

    Written as an AR7-format CSV and read with the pipeline's own reader, so sigmas are
    derived exactly as in the AR7 calibration.
    """
    ar7 = pd.read_csv(AR7_TARGETS)
    ar7 = ar7[~ar7.Variable.isin(["TCRE", "Atmospheric Concentrations|CO2"])].copy()
    gm = ar7.Variable == "Global Mean Surface Temperature (GMST)"
    ar7.loc[gm, "Variable"] = SAT
    if model is not None:
        tg = pd.read_csv(TARGETS / model / "targets.csv").iloc[0]  # truth member first
        for var, val in [(SAT, tg.GMST_2004_2023), ("Ocean Heat Content|Global|Total", tg.OHC_1971_2020_ZJ)]:
            r = ar7.Variable == var
            shift = val - ar7.loc[r, "50th"].iloc[0]
            ar7.loc[r, ["5th", "50th", "95th"]] += shift
    d = RUNS / (model or "prior")
    d.mkdir(parents=True, exist_ok=True)
    ar7.to_csv(d / "constraints_ar7_format.csv", index=False)
    cc = shf.make_constraints_config_from_RCMIP_csv(d / "constraints_ar7_format.csv")
    cc.loc[cc["Variable Name"] == SAT, "Varname_short"] = "GMST_SAT"
    return cc


def prior_calibdata():
    cc = constraints(None)
    return cc[~cc["Variable Name"].isin(["ECS", "TCR"])].reset_index(drop=True)


# ------------------------------------------------------------------ stages
def stage_prior(n):
    set_run_dir(RUNS / "prior")
    cfg = json.load(open(PRIOR_CONFIG))["prior_configs"]
    assert "pdo_timeseries_path" not in cfg and "delta_lambda_pdo" not in cfg["prior_distro_dict"]
    # parameters switched off in a variant are given as scalars in prior_distro_dict: fix them
    dd = {k: v for k, v in cfg["prior_distro_dict"].items() if isinstance(v, list)}
    sv = {**cfg["set_values"], **{k: v for k, v in cfg["prior_distro_dict"].items() if not isinstance(v, list)}}
    print(f"variant {VARIANT} ({PRIOR_CONFIG.name}); fixed: {sv}", flush=True)
    testconfig = _ConfigDistro(distro_dict=dd, setvalues=sv)
    rpe.run_prior_ensemble(
        testconfig=testconfig, scenariodata=scendata("ssp245", 2023), calibdata=prior_calibdata(),
        prunecfgs={"temp": [SAT, None, 0.17]}, distnums=n, chunk_size=CHUNK,
        max_workers=WORKERS, continue_from_existing=True,
    )
    json.dump({"n": n, "variant": VARIANT, "config": PRIOR_CONFIG.name},
              open(RUNS / "prior" / "prior_meta.json", "w"))


def prior_n():
    return json.load(open(RUNS / "prior" / "prior_meta.json"))["n"]


def link_prior(out):
    src = RUNS / "prior" / "output"
    for f in src.iterdir():
        if "_chunk_" in f.name and not (out / f.name).exists():
            (out / f.name).symlink_to(f)


def skewnorm_fixed(samples, constraint_config, varnum):
    """Skew-normal with the 5th/50th/95th percentiles of the target (sigma * 1.645 either side)."""
    import scipy.optimize
    import scipy.stats

    name = constraint_config["Varname_short"][varnum]
    lo, hi = constraint_config["lower_sigma"][varnum], constraint_config["upper_sigma"][varnum]
    c = constraint_config["Central Value"][varnum]
    if lo == hi:
        samples[name] = scipy.stats.norm.rvs(loc=c, scale=lo, size=10**5, random_state=43178)
        return samples
    k = shf.SIGMA_TO_90PERCENT
    x = scipy.optimize.root(wgt.opt, [1, c, (lo + hi) * k / 2], args=(c - k * lo, c, c + k * hi)).x
    samples[name] = scipy.stats.skewnorm.rvs(x[0], loc=x[1], scale=x[2], size=10**5, random_state=19387)
    return samples


IDEAL = {"Experiments": ["abrupt-4xCO2", "1pctCO2"], "Varname_short": ["ECS", "TCR"],
         "Yearstart_norm": [1850, 1850], "Yearend_norm": [1850, 1850],
         "Yearstart_change": [1850, 1920], "Yearend_change": [2000, 1920]}
LOOKUP = RUNS / "idealised" / "ecs_tcr.csv"


def suffix_for(fixed_skew, hist_only=False):
    return ("_histonly" if hist_only else "") + ("_fixedskew" if fixed_skew else "")


def prune_model(model):
    """Prune the shared prior to the model's pseudo-observations (once); returns the output dir."""
    n = prior_n()
    k = int(os.environ.get("PM_USE_CHUNKS", int(np.ceil(n / CHUNK))))
    out = set_run_dir(RUNS / model)
    meta = RUNS / model / "prune_meta.json"
    if meta.exists():
        old = json.load(open(meta))["chunks"]
        if old != k:
            raise SystemExit(f"{model}: pruned with {old} chunks, PM_USE_CHUNKS={k}; move the directory first")
    link_prior(out)
    if not (out / "data_all_targs_paramats.h5").exists():
        prn.prune_all_chunks(
            total_samples=n, num_chunks=k, file_endstring="",
            prune_lists=[["temp", str(TARGETS / model / "pseudo_obs.csv"), 0.17]],
        )
        json.dump({"chunks": k, "n_prior": n}, open(meta, "w"))
    return out


def load_lookup():
    if LOOKUP.exists():
        return pd.read_csv(LOOKUP, index_col="sample_id")
    return pd.DataFrame(columns=["ECS", "TCR"], index=pd.Index([], name="sample_id"))


def harvest(run_out):
    """Collect ECS/TCR per sample id from a directory where rex ran (row i of chunk c = id c*10000+i)."""
    ids = np.load(run_out / "valid_sample_ids_all_chunks.npy", allow_pickle=True)
    files = sorted(glob.glob(str(run_out / "data_postpruned_idealised_experiments_chunk_*.h5")),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    nch = int(np.ceil(len(ids) / CHUNK))
    assert len(files) == nch, f"{run_out}: {len(files)} of {nch} idealised chunks"
    parts = []
    for c, f in enumerate(files):
        t = pd.read_hdf(f, "targ").sort_index()
        sid = ids[c * CHUNK:(c + 1) * CHUNK]
        assert len(t) == len(sid) and (t.index.values == np.arange(len(sid))).all(), f
        parts.append(t[["ECS", "TCR"]].set_index(pd.Index(sid, name="sample_id")))
    new = pd.concat(parts)
    lk = load_lookup()
    lk = pd.concat([lk, new[~new.index.isin(lk.index)]])
    LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    lk.to_csv(LOOKUP)
    print(f"ecs_tcr.csv: +{len(new)} from {run_out} -> {len(lk)} members", flush=True)


def stage_idealised(models):
    lk = load_lookup()
    ids_l, pm_l = [], []
    for m in models:
        out = prune_model(m)
        ids = np.load(out / "valid_sample_ids_all_chunks.npy", allow_pickle=True)
        pm = pd.read_hdf(out / "data_all_targs_paramats.h5", "parammat")
        keep = ~pd.Index(ids).isin(lk.index)
        print(f"{m}: {len(ids)} pruned, {keep.sum()} without ECS/TCR", flush=True)
        ids_l.append(ids[keep])
        pm_l.append(pm[keep])
    ids = np.concatenate(ids_l)
    pm = pd.concat(pm_l).reset_index(drop=True)
    first = ~pd.Index(ids).duplicated()
    ids, pm = ids[first], pm[first].reset_index(drop=True)
    print(f"union: {len(ids)} members to run", flush=True)
    if len(ids) == 0:
        return
    out = set_run_dir(RUNS / "idealised" / f"batch_{len(lk)}")  # deterministic name -> resumable
    if not (out / "valid_sample_ids_all_chunks.npy").exists():
        np.save(out / "valid_sample_ids_all_chunks.npy", ids)
        with pd.HDFStore(out / "data_all_targs_paramats.h5") as st:
            st["parammat"] = pm
            st["targ"] = pd.DataFrame(index=pm.index)
    rex.read_pruned_and_run_per_chunks(
        config={"prior_configs": {"max_workers": WORKERS}},
        scenario_data=idealised_scendata(IDEAL["Experiments"], IDEAL["Yearend_change"]),
        calibration_data=IDEAL, idealised_experiments=IDEAL["Experiments"],
        file_endstring="", continue_from_existing=True,
    )
    harvest(out)


def stage_calibrate(model, fixed_skew, hist_only=False, own_idealised=False):
    out = prune_model(model)
    cc = constraints(model)
    suffix = suffix_for(fixed_skew, hist_only)
    if hist_only:
        cc = cc[~cc["Variable Name"].isin(["ECS", "TCR"])].reset_index(drop=True)
        tfile = "data_all_targs_paramats.h5"
    else:
        tfile = "data_all_targs_paramats_include_idealised.h5"
        ids = np.load(out / "valid_sample_ids_all_chunks.npy", allow_pickle=True)
        lk = load_lookup()
        missing = (~pd.Index(ids).isin(lk.index)).sum()
        if missing and not own_idealised:
            print(f"{model}: {missing} pruned members lack ECS/TCR in the lookup; running them for this model")
            own_idealised = True
        if own_idealised:
            if not (out / tfile).exists():
                rex.read_pruned_and_run_per_chunks(
                    config={"prior_configs": {"max_workers": WORKERS}},
                    scenario_data=idealised_scendata(IDEAL["Experiments"], IDEAL["Yearend_change"]),
                    calibration_data=IDEAL, idealised_experiments=IDEAL["Experiments"],
                    file_endstring="", continue_from_existing=True,
                )
        else:  # ECS/TCR from the shared lookup, in the order of the pruned members
            with pd.HDFStore(out / "data_all_targs_paramats.h5", "r") as st:
                targ, pm = st["targ"], st["parammat"]
            et = lk.loc[ids, ["ECS", "TCR"]].reset_index(drop=True)
            targ = pd.concat([targ.reset_index(drop=True), et], axis=1)
            with pd.HDFStore(out / tfile, "w") as st:
                st["parammat"] = pm
                st["targ"] = targ
    print(cc.to_string())
    if fixed_skew:
        wgt.add_entry_to_sample_distributions = skewnorm_fixed
    if suffix:  # the weighting reads the pruned ids with the suffix
        ids_f = out / f"valid_sample_ids_all_chunks{suffix}.npy"
        if not ids_f.exists():
            ids_f.symlink_to(out / "valid_sample_ids_all_chunks.npy")
    wgt.weight_ensemble_and_draw(
        constraint_config=cc, file_endstring=suffix, output_ensemble_size=N_DRAW,
        plot_pam_distributions=False, targ_paramat_fname=tfile,
    )


def stage_project(model, fixed_skew, hist_only=False):
    suffix = suffix_for(fixed_skew, hist_only)
    out = set_run_dir(RUNS / model)
    pdir = RUNS / model / f"projections{suffix}"
    pdir.mkdir(exist_ok=True)
    dr = DistributionRun(None, json_file_name=str(out / f"draw_samples_{N_DRAW}{suffix}.json"), numvalues=N_DRAW)
    for scen in SSPS:
        f = pdir / f"{scen}.csv"
        if f.exists():
            continue
        res = dr.run_over_distribution(scendata(scen, 2100), PROJ_VARS, max_workers=WORKERS)
        res.to_csv(f, index=False)
        print(f"{model} {scen}: {res.run_id.nunique()} members -> {f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prior", "idealised", "harvest", "calibrate", "project"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--fixed-skew", action="store_true")
    ap.add_argument("--hist-only", action="store_true")
    ap.add_argument("--own-idealised", action="store_true", help="run ECS/TCR for this model only")
    a = ap.parse_args()
    if a.stage == "prior":
        stage_prior(a.n)
    elif a.stage == "idealised":
        stage_idealised(a.models)
    elif a.stage == "harvest":
        for m in a.models:
            harvest(RUNS / m / "output")
    elif a.stage == "calibrate":
        stage_calibrate(a.models[0], a.fixed_skew, a.hist_only, a.own_idealised)
    else:
        stage_project(a.models[0], a.fixed_skew, a.hist_only)

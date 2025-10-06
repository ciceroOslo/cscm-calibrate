import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datadir = "../../data/calibration_data_Sep2025/"


def read_noaa_gml_ml_means(timeres):
    timeres_dict = {
        "year": ["annmean", 37, ["year", "mean"]],
        "month": ["mm", 38, ["decimal", "average"]],
    }
    data = pd.read_csv(
        f"{datadir}/co2_{timeres_dict[timeres][0]}_gl.csv",
        skiprows=timeres_dict[timeres][1],
    )
    return data[timeres_dict[timeres][2]].values.transpose()


def read_gcb_data():
    data_xls = pd.read_excel(
        f"{datadir}/Global_Carbon_Budget_2024_v1.02.xlsx",
        sheet_name="Historical Budget",
        skiprows=15,
    ).fillna(0.0)
    data_xls["emissions_tot"] = (
        data_xls["fossil emissions excluding carbonation"]
        + data_xls["land-use change emissions"]
        + data_xls["cement carbonation sink"]
    )
    data_xls["fossil emissions"] = (
        data_xls["fossil emissions excluding carbonation"]
        + data_xls["cement carbonation sink"]
    )
    return data_xls


def read_gcb_ocean_carbon_data():
    data_xls = pd.read_excel(
        f"{datadir}/Global_Carbon_Budget_2024_v1.02.xlsx",
        sheet_name="Ocean Sink",
        skiprows=31,
        nrows=65,
    ).fillna(0.0)
    data_xls = data_xls.loc[:, ~data_xls.columns.str.contains("^Unnamed")]
    return data_xls


def pam_plotting(parammat, weights=None, name_epithet=""):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(30, 30))
    for i, param in enumerate(parammat.columns):
        axnow = axs[i // 5, i % 5]
        if weights is None:
            axnow.hist(parammat[param].to_numpy(), bins=50)
        else:
            axnow.hist(parammat[param].to_numpy(), weights=weights, bins=10)
        axnow.set_title(param)
    fig.suptitle(f"Parameter distributions {name_epithet}")
    fig.savefig(f"parameter_distribtutions_{name_epithet}.png")
    plt.clf()


def get_data_for_plots():

    temp_data = pd.read_csv(f"{datadir}annual_averages.csv")
    co2_conc = read_noaa_gml_ml_means("year")
    data_gcb = read_gcb_data()
    data_aer_best = pd.read_csv(
        "../../data/calibration_data_Sep2025/ERF_best_1750-2024.csv"
    )
    data_aer_5 = pd.read_csv(f"{datadir}ERF_p05_aggregates_1750-2024.csv")
    data_aer_95 = pd.read_csv(f"{datadir}ERF_p95_aggregates_1750-2024.csv")
    data_ohc = pd.read_csv(
        f"{datadir}AR6_OHC_ensemble_IGCC_update_2024-04-13.csv", skiprows=[0]
    )
    print(data_aer_5.shape)
    print(data_aer_best.shape)
    print(data_aer_95.shape)
    print(data_ohc.head())
    data_ohc.columns = data_ohc.columns.str.strip()
    print(data_ohc.columns)
    print(data_ohc.shape)
    print(data_ohc.head())
    return (
        temp_data,
        co2_conc,
        data_gcb,
        data_aer_best,
        data_aer_5,
        data_aer_95,
        data_ohc,
    )


# sys.exit(4)


def plot_distributions(results, name_epithet):
    years = results.columns[7:].to_numpy(int)
    for variable, group in results.groupby("variable"):
        fig = plt.subplot()
        data = group.iloc[:, 7:].to_numpy(float)
        # print(type(data))
        # print(data)
        # print(np.sum(np.isnan(data)))
        # print(data.shape)
        (
            temp_data,
            co2_conc,
            data_gcb,
            data_aer_best,
            data_aer_5,
            data_aer_95,
            data_ohc,
        ) = get_data_for_plots()
        if variable == "Heat Content|Ocean":
            shift = data[:, 221:223].mean(axis=1)
            data = (data.transpose() - shift).transpose()
        elif variable == "Surface Air Ocean Blended Temperature Change":
            shift = np.mean(data[:, 100:150], axis=1)
            data = (data.transpose() - shift).transpose()
        fig.plot(years, np.median(data, axis=0), label="median")
        fig.plot(years, np.mean(data, axis=0), label="mean")
        fig.plot(years, np.max(data, axis=0), label="max")
        fig.plot(years, np.min(data, axis=0), label="min")
        if variable == "Surface Air Ocean Blended Temperature Change":
            fig.plot(
                temp_data["time"].to_numpy(),
                temp_data["GMST"].to_numpy(),
                "k",
                label="OBS",
            )
        if variable == "Atmospheric Concentrations|CO2":
            fig.plot(co2_conc[0], co2_conc[1], "k", label="NOAA")
        if variable == "Ocean carbon flux":
            fig.plot(
                data_gcb["Year"].to_numpy(),
                data_gcb["ocean sink"].to_numpy(),
                "k",
                label="GCB",
            )
        if variable == "Biosphere carbon flux":
            fig.plot(
                data_gcb["Year"].to_numpy(),
                data_gcb["land sink"].to_numpy(),
                "k",
                label="GCB",
            )
        if variable == "Effective Radiative Forcing|Aerosols":
            fig.plot(
                data_aer_best["time"],
                data_aer_best["aerosol-radiation_interactions"]
                + data_aer_best["aerosol-cloud_interactions"],
                "k",
                label="IGCC",
            )
            fig.fill_between(
                data_aer_best["time"],
                data_aer_5["aerosol-radiation_interactions"]
                + data_aer_5["aerosol-cloud_interactions"],
                data_aer_95["aerosol-radiation_interactions"]
                + data_aer_95["aerosol-cloud_interactions"],
                alpha=0.5,
                label="IGCC spread",
            )
            # print(np.percentile(data, 5, axis=1))
        if variable == "Heat Content|Ocean":
            fig.plot(
                data_ohc["Year"], data_ohc["Central Estimate Full-depth"], "k", label=""
            )
            fig.axhline()
            fig.fill_between(
                data_ohc["Year"],
                data_ohc["Central Estimate Full-depth"]
                - 1.645 * data_ohc["Full-depth Uncertainty (1-sigma)"],
                data_ohc["Central Estimate Full-depth"]
                + 1.645 * data_ohc["Full-depth Uncertainty (1-sigma)"],
                alpha=0.5,
                label="IGCC spread",
            )
        fig.fill_between(
            years,
            np.percentile(data, 5, axis=0),
            np.percentile(data, 95, axis=0),
            label="5-95th percetile",
            alpha=0.5,
        )
        fig.set_xlabel("Years")
        fig.set_title(variable)
        fig.set_ylabel(group.iloc[0, 6])
        fig.legend()
        plt.savefig(f"{variable.replace(' ','').replace('|', '')}_{name_epithet}.png")
        plt.clf()

import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm, get_df_from_input_w_data_handler

#cscm_root = "../ciceroscm/"
cscm_root = "/div/no-backup-nac/users/masan/GRAFITE/ciceroscm/"
# cscm_root = "/div/no-backup/git-repos/ciceroscm/"
# Adding location of source code to system path
# os.path.dirname(__file__) gives the directory of
# current file. Put in updated path if running script from elsewhere
# os.path joins all the folders of a path together in a
# system independent way (i.e. will work equally well on Windows, linux etc)
sys.path.append(f"{cscm_root}src")
sys.path.append(f"{cscm_root}venv/lib/python3.11/site-packages/")
print(sys.path)

from ciceroscm.parallel.distributionrun import DistributionRun

special_scen_skip = ["1pctCO2-bgc", "1pctCO2-rad", "scen7-LC", "scen7-HLC", "scen7-MLC", "scen7-LNC", "scen7-MC", "esm-scen7-L", "esm-scen7-HL", "esm-scen7-ML", "esm-scen7-LN", "esm-scen7-M"]
special_mapping = {"hist":"historical", "hist-cmip6": "historical-cmip6"}

def check_if_inspected(scenario_name):
    if not os.path.exists("scenarios_inspected.txt"):
        return False
    with open("scenarios_inspected.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(scenario_name + ","):
            return True
    return False

def make_dataframe_of_zeros(varname, start_year, end_year):
    years = np.arange(start_year, end_year + 1)
    data = np.zeros(len(years))
    df = pd.DataFrame(data=data, index=years, columns=[varname])
    df.index.name="year"
    return df

# split each subframe by Scenario prefix groups
def _split_by_scenario_prefix(subdf):
    s = subdf['Scenario'].astype(str).str.lower().str.strip()
    mask_esm_all = s.str.startswith('esm-allghg')
    mask_esm = s.str.startswith('esm-') & ~mask_esm_all
    mask_other = ~s.str.startswith('esm-')
    return subdf[mask_esm_all].copy(), subdf[mask_esm].copy(), subdf[mask_other].copy()


def load_and_process_protocol(excel_file_path):
    # Load the protocol CSV file into a DataFrame
    df = pd.read_excel(excel_file_path, sheet_name="scenario_info", 
                                skiprows=[1, 2], usecols=[1, 2, 5, 6])
    #print(df.head())
    #print(df.shape)
    # normalise Type values and split into two DataFrames
    _type_norm = df['Type'].astype(str).str.lower().str.replace(r'[\s\-_]+', '', regex=True)
    df_idealised = df[_type_norm.isin(['idealised', 'idealized'])].copy()
    df_non_idealised = df[_type_norm.isin(['nonidealised', 'nonidealized', 'nonideal'])].copy()
    #print(f"idealised: {df_idealised.shape}, non-idealised: {df_non_idealised.shape}")

    df_idealised_esm_all, df_idealised_esm_other, df_idealised_other = _split_by_scenario_prefix(df_idealised)
    df_non_idealised_esm_all, df_non_idealised_esm_other, df_non_idealised_other = _split_by_scenario_prefix(df_non_idealised)

    #print("idealised      :", df_idealised.shape, "-> esm_all:", df_idealised_esm_all.shape, "esm_other:", df_idealised_esm_other.shape, "other:", df_idealised_other.shape)
    #print("non-idealised  :", df_non_idealised.shape, "-> esm_all:", df_non_idealised_esm_all.shape, "esm_other:", df_non_idealised_esm_other.shape, "other:", df_non_idealised_other.shape)
    return {
        'idealised': {
            'esm_all': df_idealised_esm_all,
            'esm_other': df_idealised_esm_other,
            'other': df_idealised_other,
        },
        'non_idealised': {
            'esm_all': df_non_idealised_esm_all,
            'esm_other': df_non_idealised_esm_other,
            'other': df_non_idealised_other,
        }
    }

def read_output_variables_from_protocol(excel_file_path):
    df = pd.read_excel(excel_file_path, sheet_name="variable_definitions", usecols=[2,6], header=0)
    #print(df.head())
    #print(df[(df["Report when"] == "Always")])
    variables_all = df[(df["Report when"] == "Always")].iloc[:,0].copy().tolist() + ["Atmospheric Concentrations|CO2"]
    variables_non_idealised  = variables_all + df[(df["Report when"] == "Non-idealised")].iloc[:,0].copy().tolist()
    variables_non_idealised_emi = variables_non_idealised + df[(df["Report when"] == "Non-idealised and if emissions-driven")].iloc[:,0].copy().tolist()
    variables_non_idealised_conc = variables_non_idealised + ["Emissions|CO2"]
    return variables_all, variables_non_idealised, variables_non_idealised_emi, variables_non_idealised_conc

ystart = 1750
yendmax= 2500
emistart = 1850
#input_dir = "/home/masan/temp/rcmip_inputs_cscm/"
input_dir = "/div/no-backup-nac/users/masan/GRAFITE/temp_indata/"
gases_ep = "gases_vupdate_2024_WMO_added_new.txt"
gases_df = get_df_from_input_w_data_handler(
    os.path.join(input_dir, gases_ep), 
    input_dir, 
    gases_ep, 
    case_type="gaspam"
    )
fallback_emissions = get_df_from_input_w_data_handler(
    None, 
    input_dir, 
    f"ssp245_em_{gases_ep}", 
    nyend=yendmax, nystart=ystart, case_type="emis"
    )
fallback_concentrations = get_df_from_input_w_data_handler(
    None, 
    input_dir, 
    f"historical_conc_{gases_ep}", 
    nyend=yendmax, nystart=ystart, case_type="conc"
    )
em_piControl = get_df_from_input_w_data_handler(
    None, 
    input_dir, 
    f"esm-picontrol_idealised_em_{gases_ep}", 
    nyend=yendmax, nystart=ystart, case_type="emis"
    )
conc_piControl = get_df_from_input_w_data_handler(
    None, 
    input_dir, 
    f"piControl_conc_{gases_ep}", 
    nyend=yendmax, nystart=ystart, case_type="conc"
    )
lucalbedo_piControl = os.path.join(input_dir,"LUCalbedo_RCMIP_constant_zero_RCMIP3.txt") 

# Not system independent, can only be run on amoc or qbo
# If running on different system, this must be changed
# to where you have ssp input files

#distrorun = DistributionRun(None, json_file_name="/div/no-backup/users/masan/SCM_stuff/subset_cscm_configfile_for_py.json")
#distrorun = DistributionRun(None, json_file_name="/div/no-backup/users/masan/SCM_stuff/subset_cscm_configfile_for_py.json")
#distrorun = DistributionRun(None, json_file_name="/div/no-backup/git-repos/ciceroscm/scripts/calib_output_2.json")
#distrorun = DistributionRun(None, json_file_name="draw_samples_500.json")
#distrorun = DistributionRun(None, json_file_name="/div/no-backup-nac/users/masan/GRAFITE/cscm-calibrate/src/cscm_calibrate/data/draw_samples_500.json")
#distrorun = DistributionRun(None, json_file_name="/div/no-backup-nac/users/masan/GRAFITE/cscm-calibrate/src/cscm_calibrate/data/draw_samples_500_w_ecs.json")
#distrorun = DistributionRun(None, json_file_name="/div/no-backup-nac/users/masan/GRAFITE/cscm-calibrate/output/draw_samples_500.json")
json_file = "../../flat10_runs_repo/draw_samples_just2.json"
json_file= "/div/no-backup-nac/users/masan/GRAFITE/cscm-calibrate/output/draw_samples_500.json"
distrorun = DistributionRun(None, json_file_name=json_file)
#distrorun = DistributionRun(None, json_file_name="/div/no-backup/users/masan/SCM_stuff/subset_cscm_configfile_for_py_small.json")
def take_scenario_row_define_scendata_and_run(row, run_type, variables=None, dont_run=False):
    #print(row)
    yend = row["Duration of scenario"] + ystart - 1
    scen_name = row["Scenario"]
    scen_name_strip = scen_name.split("esm-")[-1].split("allGHG-")[-1]
    if scen_name_strip.startswith("scen7") and scen_name_strip.endswith("C"):
        scen_name_strip = scen_name_strip[:-1]
    if run_type == "esm-allghg":
        emistart = 1850
    else:
        emistart = yend
    arg_dict = {
        "gaspam" : gases_df,
        "nyend": yend,
        "nystart": ystart,
        "emstart": emistart,
    }

    # Deal with natural forcings:
    if row["Type"] == "idealised":
        arg_dict["sunvolc"] = 0
        arg_dict["rf_luc_file"] = lucalbedo_piControl
        arg_dict["df_nat_ch4"] = make_dataframe_of_zeros("CH4", ystart, yend)
        arg_dict["df_nat_n2o"] = make_dataframe_of_zeros("N2O", ystart, yend)
    else:
        arg_dict["sunvolc"] = 1
        if os.path.exists(os.path.join(input_dir, f"solar_RCMIP_{scen_name_strip}_RCMIP3.txt")):
            arg_dict["rf_solar_file"] = f"solar_RCMIP_{scen_name_strip}_RCMIP3.txt"
        elif os.path.exists(os.path.join(input_dir, f"solar_RCMIP_{scen_name_strip.split('-')[0]}_RCMIP3.txt")):
            arg_dict["rf_solar_file"] = f"solar_RCMIP_{scen_name_strip.split('-')[0]}_RCMIP3.txt"
        elif scen_name.startswith("methanemip") and os.path.exists(os.path.join(input_dir, f"solar_RCMIP_ssp245_RCMIP3.txt")):
            arg_dict["rf_solar_file"] = f"solar_RCMIP_ssp245_RCMIP3.txt"
        else:
            # TODO check if this is appropriate in all cases
            arg_dict["rf_solar_file"] = "solar_RCMIP_historical_RCMIP3.txt"
        if os.path.exists(os.path.join(input_dir, f"VOLC_RCMIP_{scen_name_strip}_RCMIP3.txt")):
            arg_dict["rf_volc_file"] = f"VOLC_RCMIP_{scen_name_strip}_RCMIP3.txt"
        elif os.path.exists(os.path.join(input_dir, f"VOLC_RCMIP_{scen_name_strip.split('-')[0]}_RCMIP3.txt")):
            arg_dict["rf_volc_file"] = f"VOLC_RCMIP_{scen_name_strip.split('-')[0]}_RCMIP3.txt"
        elif scen_name.startswith("methanemip") and os.path.exists(os.path.join(input_dir, f"VOLC_RCMIP_ssp245_RCMIP3.txt")):
            arg_dict["rf_volc_file"] = f"VOLC_RCMIP_ssp245_RCMIP3.txt"
        else:
            # TODO check if this is appropriate in all cases
            arg_dict["rf_volc_file"] = "VOLC_RCMIP_historical_RCMIP3.txt"
        if os.path.exists(os.path.join(input_dir, f"LUCalbedo_RCMIP_{scen_name_strip}_RCMIP3.txt")):
            arg_dict["rf_luc_file"] = f"LUCalbedo_RCMIP_{scen_name_strip}_RCMIP3.txt"
        elif os.path.exists(os.path.join(input_dir, f"LUCalbedo_RCMIP_{scen_name_strip.split('-')[0]}_RCMIP3.txt")):
            arg_dict["rf_luc_file"] = f"LUCalbedo_RCMIP_{scen_name_strip.split('-')[0]}_RCMIP3.txt"
        elif scen_name.startswith("methanemip") and os.path.exists(os.path.join(input_dir, f"LUCalbedo_RCMIP_ssp245_RCMIP3.txt")):
            arg_dict["rf_luc_file"] = f"LUCalbedo_RCMIP_ssp245_RCMIP3.txt"
        elif scen_name == "esm-allGHG-scen7-H-CH4L_rcmip_draw_samples_500.csv":
            arg_dict["rf_luc_file"] = "LUCalbedo_RCMIP_scen7-H_RCMIP3.txt"
        elif scen_name == "esm-allGHG-scen7-L-CH4H_rcmip_draw_samples_500.csv":
            arg_dict["rf_luc_file"] = "LUCalbedo_RCMIP_scen7-L_RCMIP3.txt"
        else:
            # TODO check if this is appropriate in all cases
            arg_dict["rf_luc_file"] = "LUCalbedo_RCMIP_historical_RCMIP3.txt"
    
    # Pick correct emissions file:
    if os.path.exists(os.path.join(input_dir, f"{scen_name}_em_{gases_ep}")):
        arg_dict["df_emis"] = f"{scen_name}_em_{gases_ep}"
    elif os.path.exists(os.path.join(input_dir, f"{scen_name_strip}_em_{gases_ep}")):
        arg_dict["df_emis"] = f"{scen_name_strip}_em_{gases_ep}"
    elif scen_name.startswith("esm-allGHG") and os.path.exists(os.path.join(input_dir, f"esm-{scen_name_strip}_em_{gases_ep}")):
        arg_dict["df_emis"] = f"esm-{scen_name_strip}_em_{gases_ep}"
    elif scen_name_strip in special_mapping and os.path.exists(os.path.join(input_dir, f"{special_mapping[scen_name_strip]}_em_{gases_ep}")):
        arg_dict["df_emis"] = f"{special_mapping[scen_name_strip]}_em_{gases_ep}"
    elif not scen_name.startswith("esm-") and row["Type"] == "idealised":
        arg_dict["df_emis"] = em_piControl
    elif scen_name != scen_name_strip:
        print(f"Emissions file missing for scenario {scen_name}")
        sys.exit(4)
    else:
        # TODO check if this is ok...
        arg_dict["df_emis"] = fallback_emissions
    #print(arg_dict["df_emis"])

    # Pick correct concentrations file:
    if os.path.exists(os.path.join(input_dir, f"{scen_name}_conc_{gases_ep}")):
        arg_dict["df_conc"] = f"{scen_name}_conc_{gases_ep}"
    elif os.path.exists(os.path.join(input_dir, f"{scen_name_strip}_conc_{gases_ep}")):
        arg_dict["df_conc"] = f"{scen_name_strip}_conc_{gases_ep}"
    elif scen_name_strip in special_mapping and os.path.exists(os.path.join(input_dir, f"{special_mapping[scen_name_strip]}_conc_{gases_ep}")):
        arg_dict["df_conc"] = f"{special_mapping[scen_name_strip]}_conc_{gases_ep}"
    elif scen_name.startswith("esm-") and row["Type"] == "idealised":
        arg_dict["df_conc"] = conc_piControl
    elif scen_name.startswith("methanemip") and os.path.exists(os.path.join(input_dir, f"ssp245_conc_{gases_ep}")):
        arg_dict["df_conc"] = f"ssp245_conc_{gases_ep}"
    elif not scen_name.startswith("esm-allGHG"):
        print(f"Concentration file missing for scenario {scen_name}")
        sys.exit(4)
    # Need to set this to correct version
    else:
        arg_dict["df_conc"] = fallback_concentrations
    
    # TODO: Deal with natural emissions of ch4 and n2
    # Run conc, esm or esm-allghg
    #print(arg_dict)
    scendata = define_scendata_for_scm(input_dir,**arg_dict)
    scendata[0]["scenname"] = scen_name

    if not scen_name.startswith("esm") and not scen_name.startswith("methanemip"):
        scendata[0]["conc_run"] = True
    if dont_run:
        with open("scenarios_inspected.txt", "a") as f:
            f.write(f"{scen_name}, {yend}, {emistart}, {ystart}, {scendata[0]['conc_run']}, {len(scendata)}\n")
        return pd.DataFrame()
    
    # if row["Type"] == "idealised" or "piControl" in scen_name:
    #     return pd.DataFrame()


    #print(variables)
    print(scen_name)
    print(row["Type"])
    print(scendata)
    #try:
    results = distrorun.run_over_distribution(
        scendata, output_vars=variables, max_workers=20
    )
    # except Exception as e:
    #     print(f"Error running scenario {scen_name}: {e}")
    #     del scendata
    #     return pd.DataFrame()
    # Figure out what variables to output
    # Run over with distrorun, save outputs and move on

    # Special hook for branch runs etc...

    #
    del scendata
    return results

dont_run = False

if __name__ == "__main__":
    #input_dir = "/div/no-backup-nac/users/masan/GRAFITE/temp_indata"
    #protocol_file = os.path.join("..", "..","rcmip-phase-3/RCMIP3_input_datafiles/", "rcmip_phase3_protocol_v1.1.0.xlsx")
    protocol_file = os.path.join(input_dir, "rcmip_phase3_protocol_v1.1.0.xlsx")
    variables_all, variables_non_idealised, variables_non_idealised_emi, variables_non_idealised_conc =  read_output_variables_from_protocol(protocol_file)
    experiment_out = {
        "idealised":{
            "esm_all": variables_all,
            "esm_other": variables_all,
            "other": variables_all,
        },
        "non_idealised":{
            "esm_all": variables_non_idealised_emi,
            "esm_other": variables_non_idealised,
            "other": variables_non_idealised_conc,
        }
    }
    skip = []
    #skip = ["esm-1pct-brch-1000PgC"]
    split_experiment_dfs = load_and_process_protocol(protocol_file)
    print(split_experiment_dfs)
    #sys.exit(4)
    for key1 in split_experiment_dfs:
        for key2 in split_experiment_dfs[key1]:
            print(f"{key1} - {key2} : {split_experiment_dfs[key1][key2].shape}")
            print("---------------------------------------------------------")
            for index,row in split_experiment_dfs[key1][key2].iterrows():
                print(row["Scenario"])
                outpath = f"out_file_dump/{row['Scenario']}_rcmip_{json_file.split('/')[-1].split('.')[0]}.csv"
                if os.path.exists(outpath):
                    print(f"Output file {outpath} already exists, skipping run")
                    continue
                print(outpath)
                #sys.exit(4)
                # if key1 == "idealised":
                #     print(f"Skipping scenario {row['Scenario']} as it's idealised")
                #     continue
                if row["Scenario"] in skip:
                    print(f"Skipping scenario {row['Scenario']} as it's in the skip list")
                    continue
                print(row["Scenario"])
                print(special_scen_skip)
                if row["Scenario"] in special_scen_skip:
                    print(f"Skipping special scenario {row['Scenario']}")
                    continue
                if dont_run:
                    if check_if_inspected(row["Scenario"]):
                        print(f"Skipping already inspected scenario {row['Scenario']}")
                        continue
                if row["Duration of scenario"] == "Variable":
                    continue
                print(row["Scenario"])
                results = take_scenario_row_define_scendata_and_run(row, key2, variables=experiment_out[key1][key2], dont_run=dont_run)
                #print(results.head())
                #sys.exit(4)
                if not results.empty:
                    results.to_csv(f"out_file_dump/{row['Scenario']}_rcmip_{json_file.split('/')[-1].split('.')[0]}.csv")
                del results
        # #sys.exit(4)
    print("Done all scenarios")
    sys.exit(4)
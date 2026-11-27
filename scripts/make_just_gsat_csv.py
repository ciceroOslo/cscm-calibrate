import sys

import pandas as pd

folder = "out_file_dump_ar7_v1"
experiments = ["H", "HL", "L", "LN", "M", "ML", "VL"]

list_combine = []
for exp in experiments:
    file_path = f"{folder}/esm-allGHG-scen7-{exp}_ar7_natforc_draw_samples_ar7_v1_500.csv"
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.columns)
    print(df["variable"].unique())
    df_gsat = df[df["variable"] == "Surface Air Temperature Change"]
    print(df_gsat.shape)
    list_combine.append(df_gsat)
combined_df = pd.concat(list_combine, ignore_index=True)
print(combined_df.shape)
combined_df.to_csv(f"{folder}/combined_gsat_scenariomip_ciceroscm.csv", index=False)
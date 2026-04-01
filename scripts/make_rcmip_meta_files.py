import os, sys
import pandas as pd

protocol_file = os.path.join("..", "..","rcmip-phase-3/RCMIP3_input_datafiles/", "rcmip_phase3_protocol_v1.1.1.xlsx")
df_template = pd.read_excel(protocol_file, sheet_name="meta_model", 
                            skiprows=[1, 2], usecols=[1,2,3,4,5,6,7,8],)
print(df_template.columns)

#config_file = "/home/masan/temp/high_warming_member_explore/draw_samples_500.json"

input = ["ciceroscm", "CICEROSCM-PY", "v2.0.2", "default", "ciceroscm v2.0.2 with default energy balance model and default carbon cycle, but parametrised impulse response timescales", "RCMIP phase3", "Marit Sandstad", "Sandstad, M. and Aamaas, B. and Johansen, A. N. and Lund, M. T. and Peters, G. P. and Samset, B. H. and Sanderson, B. M. and Skeie, R. B. (2024) CICERO Simple Climate Model (CICERO-SCM v1.1.1)   an improved simple climate model with a parameter calibration tool. GMD 17, 2025. https://doi.org/10.5194/gmd-17-6589-2024"]

pd.DataFrame([input], columns=df_template.columns).to_csv("meta_model_ciceroscm.csv", index=False)

df_template_comments = pd.read_excel(protocol_file, sheet_name="Comments", 
                            skiprows=[1, 2], usecols=[2,3,4,5,6,7,8,9],)
print(df_template_comments.columns)
input = ["Marit Sandstad","No comment",  "marit.sandstad@cicero.oslo.no", "ciceroscm", "all", "World", "all", "all"]
pd.DataFrame([input], columns=df_template_comments.columns).to_csv("comments_ciceroscm.csv", index=False)
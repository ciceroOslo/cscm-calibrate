import os
import json
import sys

import numpy as np
import pandas as pd

output_dir = "../output/"

with open(os.path.join(output_dir, f"draw_samples_500.json"), "r") as f:
    drawn = [entry["Index"] for entry in json.load(f)]
print(drawn)
#sys.exit(0)

store = pd.HDFStore(
    os.path.join(output_dir, f"data_all_targs_paramats.h5")
)
targ = store["targ"]
parammat = store["parammat"]

store.close()
input_ensemble_size = targ.shape[0]
print(targ.shape)
print(type(targ))
print(targ.columns)
for i in range(targ.shape[0]//10000):
    print(targ.index[i*10000:(i)*10000+50])


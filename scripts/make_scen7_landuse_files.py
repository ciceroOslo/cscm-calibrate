"""Write land-use albedo forcing files for the CMIP7 scen7 tiers from the RCMIP bundle.

Each file holds albedo + irrigation ERF (W/m^2), one value per year 1750-2500, as
read by ciceroscm's ``rf_luc_file``. The sources are in the RCMIP phase 3 v3.0.1
bundle (https://doi.org/10.5281/zenodo.23347317), under
``input_datafiles_generation/data/``.

Usage: python make_scen7_landuse_files.py <bundle_dir> <output_dir>
"""

import os
import sys

import pandas as pd

TIERS = ("VL", "LN", "L", "ML", "M", "H", "HL")
SOURCES = ("Forcing_AFOLU_CO2.csv", "Forcing_irrigation_population_scale.csv")


def landuse_forcing(bundle_dir):
    """Return albedo + irrigation ERF per tier, indexed by year."""
    data_dir = os.path.join(bundle_dir, "input_datafiles_generation", "data")
    albedo, irrigation = (
        pd.read_csv(os.path.join(data_dir, f), index_col=0) for f in SOURCES
    )
    return albedo[list(TIERS)] + irrigation[list(TIERS)]


def main(bundle_dir, output_dir):
    """Write one LUCalbedo_RCMIP_scen7-<tier>_RCMIP3.txt per tier."""
    forcing = landuse_forcing(bundle_dir)
    for tier in TIERS:
        path = os.path.join(output_dir, f"LUCalbedo_RCMIP_scen7-{tier}_RCMIP3.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{v} \n" for v in forcing[tier])
        print(f"wrote {path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

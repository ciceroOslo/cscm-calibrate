from datetime import date
from pathlib import Path

import pandas as pd

s_pr_yr = 3600*24*365
sea_surface_fraction = 0.61*0.5+0.81*0.5
earth_surface = 5.101e14
conv_factor = s_pr_yr*sea_surface_fraction*earth_surface/1.e21
#print(conv_factor)



def move_ensemble_member_after_unit(df: pd.DataFrame) -> pd.DataFrame:
	"""Rename run_id to ensemble_member and place it after the unit column."""
	if "run_id" in df.columns:
		df = df.rename(columns={"run_id": "ensemble_member"})

	if "ensemble_member" not in df.columns:
		raise ValueError("Missing required column: run_id/ensemble_member")
	if "unit" not in df.columns:
		raise ValueError("Missing required column: unit")

	cols = df.columns.tolist()
	cols.remove("ensemble_member")
	unit_idx = cols.index("unit")
	cols.insert(unit_idx + 1, "ensemble_member")
	return df[cols]


def build_output_name(input_name: str, today_str: str) -> str:
	return input_name.replace("draw_samples_delta_aero_and_efficacy_wide_lambda_500", f"ciceroscm_{today_str}")

def convert_heat_uptake(df: pd.DataFrame) -> pd.DataFrame:
	"""Convert heat uptake rows to ZJ/yr using conv_factor for year columns."""
	if "variable" not in df.columns:
		return df
	if "unit" not in df.columns:
		return df

	heat_mask = df["variable"] == "Heat Uptake"
	if not heat_mask.any():
		return df

	year_cols = [col for col in df.columns if len(str(col)) == 4 and str(col).isdigit()]
	if df.loc[heat_mask, "unit"] != "ZJ/yr":
		df.loc[heat_mask, "unit"] = "ZJ/yr"
		for col in year_cols:
			df.loc[heat_mask, col] = pd.to_numeric(df.loc[heat_mask, col], errors="coerce") * conv_factor

	return df

def rename_pg_c_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Rename PgC unit to GtC."""
    if "unit" not in df.columns:
        return df

    pg_mask = df["unit"] == "Pg_C / yr"
    df.loc[pg_mask, "unit"] = "Gt C/yr"
    return df

def main(delete_after = False, convert_heat_uptake = False, resume = True) -> None:
	script_dir = Path(__file__).resolve().parent
	input_dir = script_dir / "out_file_dump"
	today_str = date.today().strftime("%Y%m%d")
	#today_str = "20260527" # For reproducibility in testing

	csv_files = sorted(input_dir.glob("*draw_samples_delta_aero_and_efficacy_wide_lambda_500.csv"))
	if not csv_files:
		print(f"No CSV files found in {input_dir}")
		return
	delete_list = []
	for csv_path in csv_files:
		output_name = build_output_name(csv_path.name, today_str)
		output_path = input_dir / output_name
		if resume and output_path.exists():
			print(f"Skipping {csv_path} because {output_path} already exists")
			continue
		df = pd.read_csv(csv_path, index_col=0)
		df = move_ensemble_member_after_unit(df)
		if convert_heat_uptake:
			df = convert_heat_uptake(df)
		df = rename_pg_c_unit(df)




		df.to_csv(output_path)
		print(f"Wrote: {output_path}")
		if delete_after:
			delete_list.append(csv_path)
	if delete_after:
		for csv_path in delete_list:
			csv_path.unlink()    
			print(f"Deleted: {csv_path}")


if __name__ == "__main__":
	main(convert_heat_uptake=False)


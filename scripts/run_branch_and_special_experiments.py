import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

from ciceroscm.parallel.cscmparwrapper import CSCMParWrapper
from ciceroscm.carbon_cycle.carbon_cycle_mod import CarbonCycleModel

from run_full_rcmip_protocol import read_output_variables_from_protocol, make_scenariodata_argdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))
from cscm_calibrate.set_up_calibration_configs_and_run import define_scendata_for_scm


INPUT_DIR = "/div/no-backup-nac/users/masan/GRAFITE/temp_indata/"
CONFIG_NAME = "draw_samples_no_efficacy_no_pattern_wide_lambda_400"
CONFIG_PATH = Path(f"../draw_samples_archive/{CONFIG_NAME}.json")


INPUT_FILENAME = f"1pctCO2_rcmip_{CONFIG_NAME}.csv"
EXCEEDANCE_THRESHOLDS_PG_C = [750, 1000, 2000]
#INPUT_DIR = "/home/masan/temp/rcmip_inputs_cscm/"

OUTPATH_MAIN = "out_file_dump_nopattern_noefficacy"

def _find_input_file() -> Path:
	"""Locate the requested input CSV, preferring the rcmip subfolder."""
	script_dir = Path(__file__).resolve().parent
	candidates = [
		script_dir / "rcmip" / INPUT_FILENAME,
		script_dir / OUTPATH_MAIN / INPUT_FILENAME,
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		f"Could not find {INPUT_FILENAME} in rcmip/ or {OUTPATH_MAIN} under {script_dir}"
	)


def _get_year_columns(df: pd.DataFrame) -> list[str]:
	return [col for col in df.columns if str(col).isdigit()]


def build_cumulative_emissions_rows(df: pd.DataFrame) -> pd.DataFrame:
	year_cols = _get_year_columns(df)
	emissions_co2 = df.loc[df["variable"] == "Emissions|CO2"].copy()
	temp_values = df.loc[df["variable"] == "Surface Air Ocean Blended Temperature Change"].copy()
	conc_values = df.loc[df["variable"] == "Atmospheric Concentrations|CO2"].copy()
	print(f"Number of Emissions|CO2 rows: {len(emissions_co2)}")

	cumulative_rows = emissions_co2.copy()
	cumulative_rows["variable"] = "Cumulative Emissions|CO2"
	cumulative_rows["unit"] = "Pg_C"
	cumulative_rows[year_cols] = cumulative_rows[year_cols].cumsum(axis=1)

	return pd.concat([emissions_co2, cumulative_rows, temp_values, conc_values], ignore_index=True)


def _first_exceedance_year(cumulative_series: pd.Series, threshold: float) -> str | None:
	for year, value in cumulative_series.items():
		if value > threshold:
			return year
	return None


def build_threshold_rows_and_exceedance_years(
	df_with_cumulative: pd.DataFrame,
	thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	year_cols = _get_year_columns(df_with_cumulative)
	new_rows: list[pd.Series] = []
	exceedance_records: list[dict] = []

	for run_id, run_df in df_with_cumulative.groupby("run_id", sort=False):
		emissions_rows = run_df.loc[run_df["variable"] == "Emissions|CO2"]
		cumulative_rows = run_df.loc[run_df["variable"] == "Cumulative Emissions|CO2"]
		if emissions_rows.empty or cumulative_rows.empty:
			continue

		emissions_row = emissions_rows.iloc[0].copy()
		cumulative_row = cumulative_rows.iloc[0][year_cols]

		for threshold in thresholds:
			exceed_year = _first_exceedance_year(cumulative_row, threshold)
			exceedance_records.append(
				{
					"run_id": run_id,
					"threshold_pg_c": threshold,
					"exceedance_year": exceed_year,
					"conclude_sumulation_year": np.min((int(exceed_year) + 500, 2500))
				}
			)

			new_row = emissions_row.copy()
			new_row["variable"] = f"Emissions|CO2|Until {int(threshold)} Pg_C"
			#new_row["threshold_pg_c"] = threshold
			#new_row["exceedance_year"] = exceed_year

			if exceed_year is None:
				new_row[year_cols] = 0.0
			else:
				exceed_idx = year_cols.index(exceed_year)
				new_row[year_cols[exceed_idx + 1 :]] = 0.0

			new_rows.append(new_row)

	threshold_rows_df = pd.DataFrame(new_rows)
	exceedance_years_df = pd.DataFrame(exceedance_records)
	return threshold_rows_df, exceedance_years_df


def get_threshold_emissions_year_values(run_group_df: pd.DataFrame, threshold: float) -> pd.Series:
	"""
	From one grouped run dataframe, return year-column values for:
	"Emissions|CO2 Until {threshold} Pg_C".
	"""
	year_cols = _get_year_columns(run_group_df)
	target_variable = f"Emissions|CO2|Until {int(threshold)} Pg_C"
	row = run_group_df.loc[run_group_df["variable"] == target_variable]

	if row.empty:
		raise ValueError(
			f"Could not find variable '{target_variable}' in grouped dataframe for run_id "
			f"{run_group_df['run_id'].iloc[0]}"
		)
	tot_values = np.zeros(751)
	tot_values[: len(year_cols)] = row.iloc[0][year_cols].values
	#sys.exit(4)
	return tot_values

def get_variable_data_as_timeseries(run_group_df: pd.DataFrame, target_variable: str) -> pd.Series:
	"""
	From one grouped run dataframe, return year-column values for:
	"Emissions|CO2 Until {threshold} Pg_C".
	"""
	year_cols = _get_year_columns(run_group_df)
	row = run_group_df.loc[run_group_df["variable"] == target_variable]

	if row.empty:
		raise ValueError(
			f"Could not find variable '{target_variable}' in grouped dataframe for run_id "
			f"{run_group_df['run_id'].iloc[0]}"
		)
	return row.iloc[0][year_cols].values.astype(float)



def get_draw_samples_entry(json_file_path: Path, run_id) -> dict | None:
	"""
	Read draw_samples_500.json and find the entry with Index equal to run_id.
	Returns the entry dict or None if not found.
	"""
	import json
	with open(json_file_path, 'r') as f:
		data = json.load(f)
	for entry in data:
		if entry.get('Index') == run_id:
			return entry
	return None


def process_runs_from_json(
		df_with_threshold_rows: pd.DataFrame, 
		json_file_path: Path, default_inputs: 
		dict, exceedance_years_df: 
		pd.DataFrame, 
		variables_list: list
		) -> None:
	"""
	Group df_with_threshold_rows by run_id and loop over them.
	For each run_id, call get_draw_samples_entry to retrieve its entry from the JSON file.
	"""
	save_data_dict = {}
	for threshold in EXCEEDANCE_THRESHOLDS_PG_C:
		save_data_dict[f"esm-1pct-brch-{threshold}PgC"] = []
	save_data_dict["1pctCO2-bgc"] = []
	save_data_dict["1pctCO2-rad"] = []
	years_1pct = _get_year_columns(df_with_threshold_rows)
	for run_id, run_df in df_with_threshold_rows.groupby("run_id", sort=False):
		entry = get_draw_samples_entry(json_file_path, run_id)
		#exceedance_rid = exceedance_years_df.loc[exceedance_years_df["run_id"] == run_id]
		if entry is not None:
			print(f"Found entry for run_id {run_id}: {entry}")
		else:
			print(f"No entry found for run_id {run_id}")
		for threshold in EXCEEDANCE_THRESHOLDS_PG_C:
			y_end = 2500

			inputs = default_inputs.copy()
			inputs[0]["nyend"] = y_end
			inputs[0]["scenname"] = f"esm-1pct-brch-{threshold}PgC"
			emissions = inputs[0]["emissions_data"]
			emissions['CO2_FF'] = get_threshold_emissions_year_values(run_df, threshold)
			parwrap = CSCMParWrapper(inputs[0])
			result = parwrap.run_over_cfgs(cfgs=[entry], output_variables = variables_list)
			save_data_dict[f"esm-1pct-brch-{threshold}PgC"].append(result)
		
		temp_data =  get_variable_data_as_timeseries(run_df, "Surface Air Ocean Blended Temperature Change")
		# print(temp_data)
		# sys.exit(4)
		conc_data = get_variable_data_as_timeseries(run_df, "Atmospheric Concentrations|CO2")
		entry["pamset_emiconc"]["nyend"] = int(years_1pct[-1])
		carbon_cycle_model = CarbonCycleModel(entry["pamset_emiconc"], pamset_carbon = entry["pamset_carbon"])
		rad_out = carbon_cycle_model.get_carbon_cycle_output(
			years_1pct, 
			conc_run= True, 
			conc_series = np.ones(len(years_1pct))*conc_data[0],
			feedback_dict_series = {"dtemp": temp_data}
			)
		save_data_dict["1pctCO2-rad"].append(format_carbon_cycle_output_for_saving(rad_out, run_id, "1pctCO2-rad", years_1pct))
		carbon_cycle_model.reset_co2_hold()
		bgc_out = carbon_cycle_model.get_carbon_cycle_output(
			years_1pct, 
			conc_run= True, 
			conc_series = conc_data,
		)
		save_data_dict["1pctCO2-bgc"].append(format_carbon_cycle_output_for_saving(bgc_out, run_id, "1pctCO2-bgc", years_1pct))
		for key, value in save_data_dict.items():
			print(f"Experiment {key} has {len(value)} runs saved.")
			print(save_data_dict[key][0].head())
	return save_data_dict

def format_carbon_cycle_output_for_saving(
		carbon_cycle_output: dict, 
		run_id:str,
		scen_name:str,
		years: np.ndarray
		) -> pd.DataFrame:
	"""
	From the carbon cycle model output dict, create a DataFrame with columns:
	"variable", "unit", "year", "value", and any other relevant metadata.
	"""
	carbon_cycle_map = {
		#"Biosphere carbon flux": ["Carbon Flux|Land", "Pg C / yr"],
		#"Ocean carbon flux": ["Carbon Flux|Ocean", "Pg C / yr"],
		#"Airborne fraction CO2": ["Airborne fraction CO2", "Unitless"],
		"Biosphere carbon pool": ["Carbon Pool|Land", "Pg C"],
		"Ocean carbon pool": ["Carbon Pool|Ocean", "Pg C"],
		"Net flux to atmosphere": ["Net Flux to Atmosphere|CO2", "Pg C / yr"],
		"Emissions": ["Emissions|CO2", "Pg C / yr"],
	}
	records = []
	for variable, meta in carbon_cycle_map.items():
		if variable in carbon_cycle_output:
			series = carbon_cycle_output[variable]
		elif variable == "Net flux to atmosphere":
			series = - carbon_cycle_output["Biosphere carbon flux"] - carbon_cycle_output["Ocean carbon flux"]
		elif variable.replace("pool", "flux") in carbon_cycle_output:
			series = np.cumsum(carbon_cycle_output[variable.replace("pool", "flux")])
		data = [
			"CICERO-SCM-PY",
			"ciceroscm",
			run_id,
			scen_name,
			"World",
			meta[0],
			meta[1],
		]
		data.extend(series.values)
		index = [
            "climate_model",
            "model",
            "run_id",
            "scenario",
            "region",
            "variable",
            "unit",
            ]
		index.extend(years)
		records.append(pd.Series(data=data, index=index))
	return pd.DataFrame(records)

def write_df_per_experiment_to_file(save_data_dict: dict) -> dict:
	"""
	From the save_data_dict, concatenate the results for each experiment into a single DataFrame.
	Returns a dict mapping experiment names to their concatenated DataFrames.
	"""
	for experiment, results in save_data_dict.items():
		experiment_df = pd.concat(results, ignore_index=True)
		experiment_df.to_csv(f"{OUTPATH_MAIN}/{experiment}_rcmip_{CONFIG_NAME}.csv", index=False)
	return 

def get_default_inputs() -> dict:
	scenname = "1pctCO2"
	scen_name_strip = "1pctCO2"
	row = {"Type": "idealised"}
	run_type = "esm"
	arg_dict = make_scenariodata_argdict(row, run_type, scenname, scen_name_strip, yend=2500)
	print(arg_dict)
	scendata = define_scendata_for_scm(INPUT_DIR, **arg_dict)
	return scendata



def main() -> None:
	input_path = _find_input_file()
	output_path = input_path.with_name(input_path.stem + "_with_cumulative.csv")
	threshold_output_path = input_path.with_name(input_path.stem + "_with_threshold_rows.csv")
	exceedance_output_path = input_path.with_name(input_path.stem + "_exceedance_years.csv")
	print(input_path)
	df = pd.read_csv(input_path)
	print(df.shape)
	df_with_cumulative = build_cumulative_emissions_rows(df)
	print(df_with_cumulative.shape)
	threshold_rows_df, exceedance_years_df = build_threshold_rows_and_exceedance_years(
		df_with_cumulative, EXCEEDANCE_THRESHOLDS_PG_C
	)
	df_with_threshold_rows = pd.concat(
		[df_with_cumulative, threshold_rows_df],
		ignore_index=True,
	)

	#df_with_cumulative.to_csv(output_path, index=False)
	#df_with_threshold_rows.to_csv(threshold_output_path, index=False)
	exceedance_years_df.to_csv(exceedance_output_path, index=False)
	print(df_with_threshold_rows.shape)
	print(df_with_threshold_rows.head())
	print(exceedance_years_df)
	print(f"Wrote output with cumulative rows to: {output_path}")
	print(f"Wrote output with threshold rows to: {threshold_output_path}")
	print(f"Wrote exceedance years to: {exceedance_output_path}")
	default_scendata = get_default_inputs()
	variables_list = read_output_variables_from_protocol(os.path.join(INPUT_DIR, "rcmip_phase3_protocol_v1.1.0.xlsx"))
	print(variables_list)
	save_data_dict = process_runs_from_json(df_with_threshold_rows, CONFIG_PATH, default_scendata, exceedance_years_df, variables_list[0])
	write_df_per_experiment_to_file(save_data_dict)



if __name__ == "__main__":
	main()
    
from datetime import date
from pathlib import Path

import pandas as pd


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
	return input_name.replace("draw_samples_just2", f"ciceroscm_{today_str}")


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	input_dir = script_dir / "out_file_dump"
	today_str = date.today().strftime("%Y%m%d")

	csv_files = sorted(input_dir.glob("*.csv"))
	if not csv_files:
		print(f"No CSV files found in {input_dir}")
		return

	for csv_path in csv_files:
		df = pd.read_csv(csv_path, index_col=0)
		df = move_ensemble_member_after_unit(df)

		output_name = build_output_name(csv_path.name, today_str)
		output_path = input_dir / output_name
		df.to_csv(output_path)
		print(f"Wrote: {output_path}")


if __name__ == "__main__":
	main()


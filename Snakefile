# Common filenames and paths -------------------------------------------------
NIS_CACHE = ".cache/nisapi"
RAW_DATA = "data/nis_raw.parquet"
FORECASTS = "data/forecasts.parquet"
SCORES = "data/scores.parquet"

import yaml
import polars as pl

# Read in workflow information from config files ------------------------------
def token():
	with open("scripts/socrata_app_token.txt") as f:
		return f.read().strip()

with open("scripts/config.yaml") as f:
	CONFIG = yaml.safe_load(f)

FORECAST_DATES = pl.date_range(start=CONFIG['forecast_dates']['start'], end=CONFIG['forecast_dates']['end'], interval=CONFIG['forecast_dates']['interval']).to_list()

# Define rules ----------------------------------------------------------------
rule score:
	"""Score all forecasts (models & forecast dates) at once, producing a single output"""
	input:
		expand("data/forecasts/model={model}/forecast_date={forecast_date}/part-0.parquet", model=CONFIG["models"], forecast_date=FORECAST_DATES),
		forecasts="data/forecasts",
		raw_data=RAW_DATA,
		script="scripts/eval.py"
	output:
		SCORES
	shell:
		"python {input.script} --forecasts={input.forecasts} --obs={input.raw_data} --output={output}"

rule forecast:
	"""Generate forecast for a single model and date"""
	input:
		raw_data=RAW_DATA,
		script="scripts/forecast.py"
	output:
		"data/forecasts/model={model}/forecast_date={forecast_date}/part-0.parquet"
	shell:
		"python {input.script} --input={input.raw_data} --model={wildcards.model} --forecast_date={wildcards.forecast_date} --output={output}"

rule raw_data:
	"""Preprocess input data"""
	input:
		cache=".cache/nisapi/clean",
		script="scripts/preprocess.py"
	output:
		RAW_DATA
	shell:
		"python {input.script} --cache={input.cache}) --output={output}"

rule cache:
	"""Cache NIS data"""
	output:
		".cache/nisapi/status.txt"
	params:
		token=token,
		cache=".cache/nisapi"
	shell:
		"python -c 'import nisapi; nisapi.cache_all_datasets({params.cache:q}, {params.token:q})'",
		"find {params.cache}/clean -type f | xargs sha1sum > {output}"

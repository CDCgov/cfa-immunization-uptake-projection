import argparse
import pickle
from pathlib import Path

import polars as pl
import yaml

# canonical column order
COLUMNS = [
    "model",
    "forecast_date",
    "season",
    "geography",
    "quantile",
    "time_end",
    "estimate",
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fits", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", help="forecasts parquet", required=True)
    args = p.parse_args()

    with open(args.fits, "rb") as f:
        models_dict = pickle.load(f)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    forecasts = pl.concat(
        [
            model.predict().with_columns(model=pl.lit(model_name)).select(COLUMNS)
            for (model_name, _), model in models_dict.items()
        ]
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    forecasts.write_parquet(args.output)

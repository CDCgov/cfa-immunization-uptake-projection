import argparse
import pickle
from pathlib import Path

import polars as pl
import yaml

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

    forecasts = []
    for (k1, k2), v in models_dict.items():
        alpha = config["pred_interval_alpha"]
        forecast = v.predict(alpha).with_columns(model=pl.lit(k1))
        forecasts.append(forecast)

    forecasts = pl.concat(forecasts)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    forecasts.write_parquet(args.output)

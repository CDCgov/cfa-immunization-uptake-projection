import argparse

import polars as pl
import yaml

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    forecast_dates = pl.date_range(
        config["forecast_dates"]["start"],
        config["forecast_dates"]["end"],
        config["forecast_dates"]["interval"],
        eager=True,
    )

    print(*forecast_dates, sep=" ")

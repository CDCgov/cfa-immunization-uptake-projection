import argparse

import polars as pl
import yaml

import iup

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--data", help="observed data", required=True)
    p.add_argument("--preds", help="predictions parquet", required=True)
    p.add_argument("--output", help="output scores parquet", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    season = config["season"]

    pred = pl.read_parquet(args.preds)
    data = pl.read_parquet(args.data)

    # score the forecasts proper, only in the season that the forecasts were made
    forecast_season = pred.select(
        iup.to_season(
            pl.col("forecast_date"),
            season_start_month=season["start_month"],
            season_start_day=season["start_day"],
            season_end_month=season["end_month"],
            season_end_day=season["end_day"],
        ).unique()
    ).to_series()
    assert len(forecast_season) == 1, "Can only score forecasts from one season"
    forecast_season = forecast_season[0]

    eos_abs_diff = iup.eos_abs_diff(
        obs=data.filter(pl.col("season") == pl.lit(forecast_season)),
        pred=pred.filter(pl.col("season") == pl.lit(forecast_season)),
        features=["season", "geography"],
    )

    eos_abs_diff.write_parquet(args.output)

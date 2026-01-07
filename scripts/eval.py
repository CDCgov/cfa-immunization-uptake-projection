import argparse

import polars as pl
import yaml

import iup.eval
import iup.utils

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--data", help="observed data", required=True)
    p.add_argument("--preds", help="predictions parquet", required=True)
    p.add_argument("--output", help="output scores parquet", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pred = pl.scan_parquet(args.preds)
    data = pl.read_parquet(args.data)

    # score each season/state fit, using all available data (i.e., only the last
    # forecast date)
    mspe = iup.eval.mspe(
        obs=data,
        pred=pred.filter(pl.col("forecast_date") == pl.col("forecast_date").max()),
        grouping_factors=config["groups"],
    )

    # score the forecasts proper, only in the season that the forecasts were made
    forecast_season = pred.select(
        pl.col("forecast_date")
        .pipe(
            iup.utils.date_to_season,
            season_start_month=config["season"]["start_month"],
            season_start_day=config["season"]["start_day"],
        )
        .unique()
    ).collect()
    assert forecast_season.height == 1
    forecast_season = forecast_season.item()

    eos_abs_diff = iup.eval.eos_abs_diff(
        obs=data.filter(pl.col("season") == pl.lit(forecast_season)),
        pred=pred.filter(pl.col("season") == pl.lit(forecast_season)),
        grouping_factors=config["groups"],
    )

    pl.concat([mspe, eos_abs_diff]).write_parquet(args.output)

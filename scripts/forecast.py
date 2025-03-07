import argparse
import datetime as dt
from typing import Any

import polars as pl
import yaml

import iup
import iup.models


def run_all_forecasts(data, config) -> pl.DataFrame:
    """Run all forecasts

    Returns:
        pl.DataFrame: data frame of forecasts, organized by model and forecast date
    """

    if config["evaluation_timeframe"]["interval"] is not None:
        forecast_dates = pl.date_range(
            config["forecast_timeframe"]["start"],
            config["forecast_timeframe"]["end"],
            config["evaluation_timeframe"]["interval"],
            eager=True,
        ).to_list()
    else:
        forecast_dates = [config["forecast_timeframe"]["start"]]

    return pl.concat(
        [
            run_forecast1(
                config_model=config_model,
                config=config,
                data=data,
                forecast_date=forecast_date,
            )
            for config_model in config["models"]
            for forecast_date in forecast_dates
        ]
    )


def run_forecast1(
    config_model: dict[str, Any],
    config: dict[str, Any],
    forecast_date: dt.date,
    data: iup.UptakeData,
):
    model_name = config_model["name"]
    model_class = getattr(iup.models, model_name)

    assert issubclass(model_class, iup.models.UptakeModel), (
        f"{model_name} is not a valid model type!"
    )

    augmented_data = model_class.augment_data(
        data=data,
        season_start_month=config["data"]["season_start_month"],
        season_start_day=config["data"]["season_start_day"],
        groups=config["data"]["groups"],
        rollouts=config["data"]["rollouts"],
    )

    train_data = iup.UptakeData.split_train_test(augmented_data, forecast_date, "train")

    # Make an instance of the model, fit it using training data, and make projections
    fit_model = model_class(config_model["seed"]).fit(
        train_data,
        config["data"]["groups"],
        config_model["params"],
        config_model["mcmc"],
    )

    # Get test data, if there is any, to know exact dates for projection
    test_data = iup.UptakeData.split_train_test(augmented_data, forecast_date, "test")
    if test_data.height == 0:
        test_data = None

    cumulative_projections = fit_model.predict(
        start_date=forecast_date,
        end_date=config["forecast_timeframe"]["end"],
        interval=config["forecast_timeframe"]["interval"],
        test_data=test_data,
        groups=config["data"]["groups"],
        season_start_month=config["data"]["season_start_month"],
        season_start_day=config["data"]["season_start_day"],
    )

    grouping_factors = config["data"]["groups"]
    if grouping_factors is None:
        grouping_factors = ["season"]

    return (
        cumulative_projections.group_by(grouping_factors + ["time_end"])
        .agg(pl.col("estimate").mean().alias("estimate"))
        .sort("time_end")
        .with_columns(
            forecast_start=forecast_date,
            forecast_end=config["forecast_timeframe"]["end"],
            model=pl.lit(model_name),
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--input", help="input data")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(pl.scan_parquet(args.input).collect())

    run_all_forecasts(input_data, config).write_parquet(args.output)

import argparse
import datetime as dt
from typing import Any, List, Type

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

    all_forecast = pl.DataFrame()

    for config_model in config["models"]:
        model_name = config_model["name"]
        model_class = getattr(iup.models, model_name)

        assert issubclass(model_class, iup.models.UptakeModel), (
            f"{model_name} is not a valid model type!"
        )

        augmented_data = model_class.augment_data(
            data,
            config["data"]["season_start_month"],
            config["data"]["season_start_day"],
            config["data"]["groups"],
            config["data"]["rollouts"],
        )

        for forecast_date in forecast_dates:
            forecast = run_forecast(
                data=augmented_data,
                model_class=model_class,
                seed=config_model["seed"],
                params=config_model["params"],
                mcmc=config_model["mcmc"],
                grouping_factors=config["data"]["groups"],
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                forecast_interval=config["forecast_timeframe"]["interval"],
                season_start_month=config["data"]["season_start_month"],
                season_start_day=config["data"]["season_start_day"],
            )

            forecast = forecast.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                model=pl.lit(model_name),
            )

            all_forecast = pl.concat([all_forecast, forecast])

    return all_forecast


def run_forecast(
    data: iup.UptakeData,
    model_class: Type[iup.models.UptakeModel],
    seed: int,
    params: dict[str, Any],
    mcmc: dict[str, Any],
    grouping_factors: List[str] | None,
    forecast_start: dt.date,
    forecast_end: dt.date,
    forecast_interval: str,
    season_start_month: int,
    season_start_day: int,
) -> pl.DataFrame:
    """Run a single model for a single forecast date"""
    train_data = iup.UptakeData.split_train_test(data, forecast_start, "train")

    # Make an instance of the model, fit it using training data, and make projections
    fit_model = model_class(seed).fit(
        train_data,
        grouping_factors,
        params,
        mcmc,
    )

    # Get test data, if there is any, to know exact dates for projection
    test_data = iup.UptakeData.split_train_test(data, forecast_start, "test")
    if test_data.height == 0:
        test_data = None

    cumulative_projections = fit_model.predict(
        start_date=forecast_start,
        end_date=forecast_end,
        interval=forecast_interval,
        test_data=test_data,
        groups=grouping_factors,
        season_start_month=season_start_month,
        season_start_day=season_start_day,
    )

    if grouping_factors is None:
        grouping_factors = ["season"]

    return cumulative_projections


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

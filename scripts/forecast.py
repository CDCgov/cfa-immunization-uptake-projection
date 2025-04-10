import argparse
import datetime as dt
import pickle
from typing import Any, Dict, List

import numpy as np
import polars as pl
import yaml

import iup
import iup.models
from iup.utils import parse_name_and_date


def run_all_forecasts(
    data: iup.UptakeData,
    fitted_models: Dict[str, iup.models.UptakeModel],
    config: dict[str, Any],
) -> pl.DataFrame:
    """Run all forecasts for all the fitted models across model name and forecast start.

    Args:
        data: iup.UptakeData
            all available data including training and testing.
        fitted_models: dict
            a dictionary containing all fitted models, indexed by a
            combo of model name and forecast start date.
        config: yaml
            config file to specify args in augment_data and run_forecast.


    Returns:
        A pl.DataFrame saving predictive distribution at each time point between
        forecast start and end, at least grouped by model name, forecast start, and forecast end.
    """

    forecast_dates_list = [
        parse_name_and_date(str)["forecast_date"] for str in fitted_models.keys()
    ]
    forecast_dates = np.array(forecast_dates_list)
    forecast_dates = np.unique(forecast_dates)
    forecast_dates.sort()

    model_names = [
        parse_name_and_date(str)["model_name"] for str in fitted_models.keys()
    ]
    all_forecast = pl.DataFrame()

    for model_name in model_names:
        assert hasattr(iup.models, model_name), (
            f"{model_name} is not a valid model type!"
        )

        model_class = getattr(iup.models, model_name)

        augmented_data = model_class.augment_data(
            data,
            config["data"]["season_start_month"],
            config["data"]["season_start_day"],
            config["data"]["groups"],
            config["data"]["rollouts"],
        )

        for forecast_date in forecast_dates:
            sel_key = [
                key
                for key in fitted_models
                if parse_name_and_date(key)["forecast_date"] == forecast_date
                if parse_name_and_date(key)["model_name"] == model_name
            ]

            model = fitted_models[sel_key[0]]

            forecast = run_forecast(
                data=augmented_data,
                fit_model=model,
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
    grouping_factors: List[str] | None,
    fit_model: iup.models.UptakeModel,
    forecast_start: dt.date,
    forecast_end: dt.date,
    forecast_interval: str,
    season_start_month: int,
    season_start_day: int,
) -> pl.DataFrame:
    """
    Given fitted model object, get forecast using predictors in test data.

    Args:
        data: iup.UptakeData
            all available data including training and testing data.
        grouping factors: List[str] | None
            A list of column names to group "estimate" (dependent variable) in the data.
        fit_model: iup.models.UptakeModel
            A single iup.models.UptakeModel that is fitted.
        forecast_start: dt.date
            The first day of forecast.
        forecast_end: dt.date
            The last day of forecast.
        season_start_month: int
            The first month of an overwinter season.
        season_start_day: int
            The first day in the first month of an overwinter season.

    Return:
        A pl.DataFrame that records a predictive distribution at each time point
        between forecast start and end from a fitted model.
    """
    # Get test data, if there is any, to know exact dates for projection
    test_data = iup.UptakeData.split_train_test(data, forecast_start, "test")
    if test_data.height == 0:
        test_dates = None
    else:
        test_dates = test_data.select(["time_end", "season"])

    cumulative_projections = fit_model.predict(
        start_date=forecast_start,
        end_date=forecast_end,
        interval=forecast_interval,
        test_dates=test_dates,
        groups=grouping_factors,
        season_start_month=season_start_month,
        season_start_day=season_start_day,
    )

    return cumulative_projections


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--input", help="input data")
    p.add_argument("--models", help="fitted models")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(pl.scan_parquet(args.input).collect())

    with open(args.models, "rb") as f:
        models = pickle.load(f)

    run_all_forecasts(input_data, models, config).write_parquet(args.output)

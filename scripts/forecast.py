import argparse
import datetime as dt
import pickle
from typing import Any, List

import numpy as np
import polars as pl
import yaml

import iup
import iup.models


def run_all_forecasts(
    data: iup.UptakeData,
    fitted_models: List[iup.models.UptakeModel],
    config: dict[str, Any],
) -> pl.DataFrame:
    """Run all forecasts

    Returns:
        dictionary of two data frames: forecasts and posterior distributions,
        both organized by model and forecast date
    """

    train_end_dates_list = [model.end_date[0, 0] for model in fitted_models]
    train_end_dates = np.array(train_end_dates_list)
    train_end_dates = np.unique(train_end_dates)
    train_end_dates.sort()

    if config["evaluation_timeframe"]["interval"] is not None:
        forecast_dates = pl.date_range(
            config["forecast_timeframe"]["start"],
            config["forecast_timeframe"]["end"],
            config["evaluation_timeframe"]["interval"],
            eager=True,
        ).to_list()
    else:
        forecast_dates = [config["forecast_timeframe"]["start"]]

    model_names = [model.__class__.__name__ for model in fitted_models]

    all_forecast = pl.DataFrame()
    all_posterior = []

    for model_name in model_names:
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

        model_posterior = pl.DataFrame()

        for i, train_end in enumerate(train_end_dates):
            # train end maps to forecast date on 1-on-1 base,
            # # can directly index forecast date using index of train end #
            # print(
            #     f"train_end: {train_end}, forecast_date: {forecast_dates[i]}, model: {model_name}"
            # )
            forecast_date = forecast_dates[i]

            fitted_model_list = [
                model
                for model in fitted_models
                if model.end_date[0, 0] == train_end
                if model.__class__.__name__ == model_name
            ]

            assert len(fitted_model_list) == 1, (
                f"More than one fitted model found for {model_name} with {train_end}"
            )

            fitted_model = fitted_model_list[0]

            forecast = run_forecast(
                data=augmented_data,
                fit_model=fitted_model,
                grouping_factors=config["data"]["groups"],
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                forecast_interval=config["forecast_timeframe"]["interval"],
                season_start_month=config["data"]["season_start_month"],
                season_start_day=config["data"]["season_start_day"],
            )

            forecast = model_output["projections"]

            forecast = forecast.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                model=pl.lit(model_name),
            )
            all_forecast = pl.concat([all_forecast, forecast])

            posterior = model_output["posterior"]
            posterior = posterior.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                model=pl.lit(model_name),
            )
            model_posterior = pl.concat([model_posterior, posterior])

        all_posterior.append(model_posterior)

    return {"forecasts": all_forecast, "posteriors": all_posterior}


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
    """Given fitted model object, get forecast using predictors in test data"""
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

    return {"posterior": posterior, "projections": cumulative_projections}


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

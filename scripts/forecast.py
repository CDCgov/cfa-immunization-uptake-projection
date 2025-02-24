import argparse

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

    for model in config["models"]:
        for forecast_date in forecast_dates:
            forecast = run_forecast(
                model,
                data,
                grouping_factors=config["data"]["groups"],
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
            )

            forecast = forecast.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                model=pl.lit(model["name"]),
            )

            all_forecast = pl.concat([all_forecast, forecast])

    return all_forecast


def run_forecast(
    model,
    data,
    grouping_factors,
    forecast_start,
    forecast_end,
) -> pl.DataFrame:
    """Run a single model for a single forecast date"""
    assert issubclass(getattr(iup.models, model["name"]), iup.models.UptakeModel), (
        f"{model['name']} is not a valid model type!"
    )

    # Format training data according to the type of model desired
    if model["name"] == "LinearIncidentUptakeModel":
        data = data.to_incident(grouping_factors)
        train_data = iup.UptakeData.split_train_test(data, forecast_start, "train")
    elif model["name"] == "Hill":
        data = iup.CumulativeUptakeData(
            data.with_columns(
                elapsed=iup.models.HillModel.date_to_elapsed(
                    pl.col("date"),
                    config["data"]["season_start_month"],
                    config["data"]["season_start_day"],
                )
            )
        )
        train_data = iup.UptakeData.split_train_test(data, forecast_start, "train")

    # Make an instance of the model, fit it using training data, and make projections
    fit_model = getattr(iup.models, model["name"])(model["seed"]).fit(
        train_data,
        grouping_factors,
        model["params"],
        model["mcmc"],
    )

    # Get test data, if there is any, to know exact dates for projection
    test_data = iup.UptakeData.split_train_test(data, forecast_start, "test")
    if test_data.height == 0:
        test_data = None

    # LEFT OFF HERE - MAY BE MISTAKES BELOW
    cumulative_projections = fit_model.predict(
        forecast_start,
        forecast_end,
        config["forecast_timeframe"]["interval"],
        test_data,
        grouping_factors,
    )

    cumulative_projections = (
        cumulative_projections.group_by(grouping_factors + ["time_end"])
        .agg(pl.col("estimate").mean().alias("estimate"))
        .sort("time_end")
    )

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

import argparse

import polars as pl
import yaml

import iup.models


def run_all_forecasts(data, config) -> pl.DataFrame:
    """Run all forecasts

    Returns:
        pl.DataFrame: data frame of forecasts, organized by model and forecast date
    """

    models = [getattr(iup.models, model_name) for model_name in config["models"]]
    assert all(issubclass(model, iup.models.UptakeModel) for model in models)

    all_forecast = pl.DataFrame()

    for model in models:
        for forecast_date in config["timeframe"]["start"]:
            forecast = run_forecast(
                model,
                data,
                grouping_factors=config["data"]["groups"],
                forecast_start=forecast_date,
                forecast_end=config["timeframe"]["end"],
            )

            forecast = forecast.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["timeframe"]["end"],
                model=pl.lit(model.__name__),
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

    # preprocess.py returns cumulative data, so convert to incident for LinearIncidentUptakeModel
    incident_data = data.to_incident(grouping_factors)

    # Prune to only the training portion
    incident_train_data = iup.IncidentUptakeData.split_train_test(
        incident_data, forecast_start, "train"
    )

    # Fit models using the training data and make projections
    fit_model = model().fit(incident_train_data, grouping_factors)

    cumulative_projections = fit_model.predict(
        forecast_start,
        forecast_end,
        config["timeframe"]["interval"],
        grouping_factors,
    )

    return cumulative_projections


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--input", help="input data")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(pl.scan_parquet(args.input).collect())

    run_all_forecasts(input_data, config).write_parquet(args.output)

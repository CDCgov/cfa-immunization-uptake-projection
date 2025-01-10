import argparse

import polars as pl
import yaml

import iup.models


def run_all_forecasts(clean_data, config) -> pl.DataFrame:
    """Run all forecasts

    Returns:
        pl.DataFrame: data frame of forecasts, organized by model and forecast date
    """
    forecast_dates = pl.date_range(
        config["timeframe"]["start"],
        config["timeframe"]["end"],
        config["timeframe"]["interval"],
        eager=True,
    )
    models = [getattr(iup.models, model_name) for model_name in config["models"]]
    assert all(issubclass(model, iup.models.UptakeModel) for model in models)

    all_forecast = pl.DataFrame()

    for model in models:
        for forecast_date in forecast_dates:
            # Get data available as of the forecast date
            forecast = run_forecast(
                model,
                clean_data,
                grouping_factors=config["groups"],
                forecast_start=config["timeframe"]["start"],
                forecast_end=forecast_date,
            )

            forecast = forecast.with_columns(
                forecast_start=config["timeframe"]["start"],
                forecast_end=forecast_date,
                model=pl.lit(model.__name__),
            )

            all_forecast = pl.concat([all_forecast, forecast])

    return all_forecast


def run_forecast(
    model,
    incident_data,
    grouping_factors,
    forecast_start,
    forecast_end,
) -> pl.DataFrame:
    """Run a single model for a single forecast date"""

    incident_train_data = iup.IncidentUptakeData(
        iup.IncidentUptakeData.split_train_test(
            incident_data, config["timeframe"]["start"], "train"
        )
    )

    # Fit models using the training data and make projections
    fit_model = model().fit(incident_train_data, grouping_factors)

    cumulative_projections = fit_model.predict(
        forecast_start,
        forecast_end,
        config["timeframe"]["interval"],
        grouping_factors,
    )

    incident_projections = cumulative_projections.to_incident(grouping_factors)

    return pl.concat(
        [
            cumulative_projections.with_columns(estimate_type=pl.lit("cumulative")),
            incident_projections.with_columns(estimate_type=pl.lit("incident")),
        ]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--input", help="input data")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = pl.scan_parquet(args.input).collect()

    all_forecast = run_all_forecasts(config=config, clean_data=input_data)
    all_forecast.write_parquet(args.output)

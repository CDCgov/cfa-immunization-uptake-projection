import argparse

import polars as pl
import yaml

def run_all_forecasts() -> pl.DataFrame:
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

    for model in models:
            for forecast_date in forecast_dates:
                # Get data available as of the forecast date


def run_forecast() -> pl.DataFrame:
    """Run a single model for a single forecast date"""
    incident_train_data = iup.IncidentUptakeData(
        iup.IncidentUptakeData.split_train_test(
            incident_data, config["timeframe"]["start"], "train"
        )
    )

    # Fit models using the training data and make projections
    fit_model = model().fit(incident_train_data, grouping_factors)

    cumulative_projections = fit_model.predict(
        config["timeframe"]["start"],
        config["timeframe"]["end"],
        config["timeframe"]["interval"],
        grouping_factors,
    )
    # save these projections somewhere

    incident_projections = cumulative_projections.to_incident(
        grouping_factors
    )
    # save these projections somewhere

    # Evaluation / Post-processing --------------------------------------------

    incident_test_data = iup.IncidentUptakeData(
        iup.IncidentUptakeData.split_train_test(
            incident_data, config["timeframe"]["start"], "test"
        )
    ).filter(pl.col("date") <= config["timeframe"]["end"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--input", help="input data")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = pl.scan_parquet(args.input)

    run_all_forecasts(config=config, cache=args.cache)

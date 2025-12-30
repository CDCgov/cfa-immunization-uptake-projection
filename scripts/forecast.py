import argparse
import datetime as dt
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import polars as pl
import yaml

import iup
import iup.models


def run_all_forecasts(
    data: iup.UptakeData,
    fitted_models: Dict[str, iup.models.UptakeModel],
    config: dict[str, Any],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
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
    all_forecasts = pl.DataFrame()
    all_postchecks = pl.DataFrame()

    for (model_name, forecast_date), fitted_model in fitted_models.items():
        assert isinstance(forecast_date, dt.date)
        assert hasattr(iup.models, model_name), (
            f"{model_name} is not a valid model type!"
        )
        model_class = getattr(iup.models, model_name)

        augmented_data = model_class.augment_data(
            data,
            config["season"]["start_month"],
            config["season"]["start_day"],
        )

        train_data, test_data = iup.UptakeData.split_train_test(
            augmented_data, forecast_date
        )
        train_dates = train_data.select(["time_end", "season"])
        if test_data.height == 0:
            test_dates = None
        else:
            test_dates = test_data.select(["time_end", "season"])

        postcheck = fitted_model.predict(
            start_date=train_data["time_end"].min(),
            end_date=train_data["time_end"].max(),
            interval=config["forecasts"]["start_date"]["interval"],
            test_dates=train_dates,
            groups=config["groups"],
            season_start_month=config["season"]["start_month"],
            season_start_day=config["season"]["start_day"],
        )

        postcheck = postcheck.with_columns(
            forecast_start=forecast_date,
            forecast_end=config["forecasts"]["end_date"],
            model=pl.lit(model_name),
        )

        all_postchecks = pl.concat([all_postchecks, postcheck])

        forecast = fitted_model.predict(
            start_date=forecast_date,
            end_date=config["forecasts"]["end_date"],
            interval=config["forecasts"]["start_date"]["interval"],
            test_dates=test_dates,
            groups=config["groups"],
            season_start_month=config["season"]["start_month"],
            season_start_day=config["season"]["start_day"],
        )

        forecast = forecast.with_columns(
            forecast_start=forecast_date,
            forecast_end=config["forecasts"]["end_date"],
            model=pl.lit(model_name),
        )

        all_forecasts = pl.concat([all_forecasts, forecast])

    return (all_postchecks, all_forecasts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--data", help="input data", required=True)
    p.add_argument("--fits", required=True)
    p.add_argument("--output", help="forecasts parquet", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(pl.read_parquet(args.data))

    with open(args.fits, "rb") as f:
        models = pickle.load(f)

    postchecks, forecasts = run_all_forecasts(input_data, models, config)

    forecasts.write_parquet(args.output)
    postchecks.write_parquet(Path(args.output).parent / "postchecks.parquet")

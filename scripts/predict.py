import argparse
import datetime as dt
import pickle
from typing import Any, Dict

import polars as pl
import yaml

import iup
import iup.models


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
    all_forecasts = pl.DataFrame()

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

        test_data = augmented_data.select(
            ["time_end", "season"] + [x for x in config["groups"] if x != "season"]
        )

        forecast = fitted_model.predict(
            test_data=test_data,
            groups=config["groups"],
            season_start_month=config["season"]["start_month"],
            season_start_day=config["season"]["start_day"],
        ).with_columns(forecast_start=forecast_date, model=pl.lit(model_name))

        all_forecasts = pl.concat([all_forecasts, forecast])

    return all_forecasts


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

    forecasts = run_all_forecasts(input_data, models, config)

    forecasts.write_parquet(args.output)

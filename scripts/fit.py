import argparse
import datetime as dt
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Type

import numpyro
import polars as pl
import yaml

import iup
import iup.models


def fit_all_models(data, config) -> Dict[str, iup.models.UptakeModel]:
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

    all_models = {}

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
        )

        for forecast_date in forecast_dates:
            fitted_model = fit_model(
                data=augmented_data,
                model_class=model_class,
                seed=config_model["seed"],
                params=config_model["params"],
                mcmc=config["mcmc"],
                grouping_factors=config["data"]["groups"],
                forecast_start=forecast_date,
            )

            label = (model_name, forecast_date)
            all_models[label] = fitted_model

    return all_models


def fit_model(
    data: iup.UptakeData,
    model_class: Type[iup.models.UptakeModel],
    seed: int,
    params: dict[str, Any],
    mcmc: dict[str, Any],
    grouping_factors: List[str] | None,
    forecast_start: dt.date,
) -> iup.models.UptakeModel:
    """fit model using training data, return fitted model object"""

    """Run a single model for a single forecast date"""
    train_data, _ = iup.UptakeData.split_train_test(data, forecast_start)

    # Make an instance of the model, fit it using training data, and make projections
    fit_model = model_class(seed).fit(
        train_data,
        grouping_factors,
        params,
        mcmc,
    )

    return fit_model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--input", help="input data directory")
    p.add_argument("--output", help="output directory")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(
        pl.scan_parquet(Path(args.input, "nis_data.parquet")).collect()
    )

    numpyro.set_host_device_count(config["mcmc"]["num_chains"])

    all_models = fit_all_models(input_data, config)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output, "model_fits.pkl"), "wb") as f:
        pkl.dump(all_models, f)

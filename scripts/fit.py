import os

# silence Jax CPU warning
os.environ["JAX_PLATFORMS"] = "cpu"

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


def fit_all_models(
    data, forecast_date: dt.date, config
) -> Dict[str, iup.models.UptakeModel]:
    """
    Run all forecasts

    Returns:
        pl.DataFrame: data frame of forecasts, organized by model and forecast date
    """

    all_models = {}

    for config_model in config["models"]:
        model_name = config_model["name"]
        model_class = getattr(iup.models, model_name)

        assert issubclass(model_class, iup.models.UptakeModel), (
            f"{model_name} is not a valid model type!"
        )

        augmented_data = model_class.augment_data(
            data,
            config["season"]["start_month"],
            config["season"]["start_day"],
        )

        fitted_model = fit_model(
            data=augmented_data,
            model_class=model_class,
            seed=config_model["seed"],
            params=config_model["params"],
            mcmc=config["mcmc"],
            grouping_factors=config["groups"],
            forecast_date=forecast_date,
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
    forecast_date: dt.date,
) -> iup.models.UptakeModel:
    """Run a single model for a single forecast date"""
    train_data = iup.UptakeData(data.filter(pl.col("time_end") <= forecast_date))

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
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--data", help="input data", required=True)
    p.add_argument("--forecast_date", required=True)
    p.add_argument("--output", help="output pickle path", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    forecast_date = dt.date.fromisoformat(args.forecast_date)
    data = iup.CumulativeUptakeData(pl.read_parquet(args.data))

    numpyro.set_host_device_count(config["mcmc"]["num_chains"])
    all_models = fit_all_models(data=data, forecast_date=forecast_date, config=config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pkl.dump(all_models, f)

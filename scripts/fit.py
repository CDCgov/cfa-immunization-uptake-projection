import os

# silence Jax CPU warning
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import datetime as dt
import pickle as pkl
from pathlib import Path
from typing import Any, Tuple

import numpyro
import polars as pl
import yaml

import iup
import iup.models


def fit_all_models(
    data: pl.DataFrame, forecast_date: dt.date, config: dict[str, Any]
) -> dict[Tuple[str, dt.date], iup.models.CoverageModel]:
    """Run all forecasts.

    Args:
        data: Input data to fit models on.
        forecast_date: Forecast date to use as training cutoff.
        config: Configuration dictionary.

    Returns:
        Dictionary of fitted models organized by model name and forecast date.
    """

    all_models = {}

    for config_model in config["models"]:
        model_name = config_model["name"]
        model_class = getattr(iup.models, model_name)

        assert issubclass(model_class, iup.models.CoverageModel), (
            f"{model_name} is not a valid model type!"
        )

        model = model_class(
            data=data,
            forecast_date=forecast_date,
            groups=config["groups"],
            seed=config_model["seed"],
            model_params=config_model["model_params"],
            mcmc_params=config["mcmc"],
        )

        model.fit()

        label = (model_name, forecast_date)
        all_models[label] = model

    return all_models


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
    data = iup.CumulativeCoverageData(pl.read_parquet(args.data))

    numpyro.set_host_device_count(config["mcmc"]["num_chains"])
    all_models = fit_all_models(data=data, forecast_date=forecast_date, config=config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pkl.dump(all_models, f)

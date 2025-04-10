import argparse
import pickle
from typing import Any, Dict  # , List

import arviz as az
import polars as pl
import yaml

import iup
import iup.diagnostics
import iup.models
from iup.utils import parse_name_and_date


def diagnostic_plot(
    models: Dict[str, iup.models.UptakeModel], config: Dict[str, Any], output_dir
):
    """select the fitted model using model name and training end date
    and generate selected diagnostic plots"""

    sel_model_dict = select_model_to_diagnose(models, config)

    diagnose_plot_names = config["diagnostics"]["plot"]

    for key, model in sel_model_dict.items():
        idata = az.from_numpyro(model.mcmc)

        for plot_name in diagnose_plot_names:
            plot_func = getattr(iup.diagnostics, plot_name)
            axes = plot_func(idata)
            fig = axes.ravel()[0].figure
            fig.savefig(f"{output_dir}/{key}_{plot_name}.png")


def diagnostic_table(
    models: Dict[str, iup.models.UptakeModel], config: Dict[str, Any], output_dir
):
    """select the fitted model using model name and training end date
    and generate selected diagnostics: summary/posterior as parquet"""

    sel_model_dict = select_model_to_diagnose(models, config)

    diagnose_table_names = config["diagnostics"]["table"]

    for key, model in sel_model_dict.items():
        idata = az.from_numpyro(model.mcmc)

        for table_name in diagnose_table_names:
            table_func = getattr(iup.diagnostics, table_name)
            if table_name == "print_posterior_dist":
                output = table_func(model, idata)
            else:
                output = table_func(idata)

            output.write_parquet(f"{output_dir}/{key}_{table_name}.parquet")


def select_model_to_diagnose(models: Dict[str, iup.models.UptakeModel], config) -> dict:
    """Select the model to diagnose based on the model name and the training end date"""

    assert len(config["diagnostics"]["forecast_date"]) <= 2, (
        "forecast_date should be None or a list of length 1 or 2"
    )

    if config["diagnostics"]["forecast_date"] is None:
        forecast_dates = pl.date_range(
            config["forecast_timeframe"]["start"],
            config["forecast_timeframe"]["end"],
            config["evaluation_timeframe"]["interval"],
            eager=True,
        )
    elif len(config["diagnostics"]["forecast_date"]) == 1:
        forecast_dates = config["diagnostics"]["forecast_date"]
    else:
        forecast_dates = pl.date_range(
            config["diagnostics"]["forecast_date"][0],
            config["diagnostics"]["forecast_date"][1],
            interval=config["evaluation_timeframe"]["interval"],
            eager=True,
        )

    sel_keys = [
        key
        for key in models.keys()
        if parse_name_and_date(key)["model_name"] in config["diagnostics"]["model"]
        if parse_name_and_date(key)["forecast_date"] in forecast_dates
    ]

    return {key: models[key] for key in sel_keys}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--input", help="fitted models")
    p.add_argument("--output_dir", help="path of output plots and tables")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.input, "rb") as f:
        models = pickle.load(f)

    diagnostic_plot(models, config, args.output_dir)
    diagnostic_table(models, config, args.output_dir)

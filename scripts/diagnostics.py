import argparse
import pickle
from typing import List

import arviz as az
import polars as pl
import yaml

import iup
import iup.diagnostics
import iup.models


#### Given fit_model.mcmc, return diagnostic plot and summary ###
def diagnostic_plot(models: List[iup.models.UptakeModel], config, output_dir):
    """select the fitted model using model name and training end date
    and generate selected diagnostic plots"""

    sel_model = select_model_to_diagnose(models, config)

    diagnose_plot_names = config["diagnostics"]["plot"]

    idata = az.from_numpyro(sel_model.mcmc)

    for plot_name in diagnose_plot_names:
        plot_func = getattr(iup.diagnostics, plot_name)
        axes = plot_func(idata)
        fig = axes.ravel()[0].figure
        fig.savefig(f"{output_dir}/{plot_name}.png")


def diagnostic_table(models: List[iup.models.UptakeModel], config, output_dir):
    """select the fitted model using model name and training end date
    and generate selected diagnostics: summary/posterior as parquet"""
    sel_model = select_model_to_diagnose(models, config)

    diagnose_table_names = config["diagnostics"]["table"]

    idata = az.from_numpyro(sel_model.mcmc)
    for table_name in diagnose_table_names:
        table_func = getattr(iup.diagnostics, table_name)
        output = pl.from_pandas(table_func(idata))
        output.write_parquet(f"{output_dir}/{table_name}.parquet")


## select the model to diagnose ##
def select_model_to_diagnose(models: List[iup.models.UptakeModel], config):
    """Select the model to diagnose based on the model name and the training end date"""
    sel_models = [
        model
        for model in models
        if model.__class__.__name__ == config["diagnostics"]["model"]
    ]

    sel_model = [
        model
        for model in sel_models
        if model.end_date[0, 0] == config["diagnostics"]["train_end"]
    ]

    assert len(sel_model) == 1, f"Expected 1 model, got {len(sel_model)}"
    sel_model = sel_model[0]

    return sel_model


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

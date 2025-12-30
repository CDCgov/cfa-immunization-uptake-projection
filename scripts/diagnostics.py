import argparse
import datetime as dt
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

import iup
import iup.diagnostics
import iup.models


def diagnostic_plot(
    models: Dict[Tuple[str, dt.date], iup.models.UptakeModel],
    config: Dict[str, Any],
    output_dir,
):
    """select the fitted model using model name and training end date
    and generate selected diagnostic plots"""

    sel_model_dict = select_model_to_diagnose(models, config)

    diagnose_plot_names = config["diagnostics"]["plot"]

    for key, model in sel_model_dict.items():
        for plot_name in diagnose_plot_names:
            plot_func = getattr(iup.diagnostics, plot_name)
            axes = plot_func(model)
            fig = axes.ravel()[0].figure
            fig.savefig(
                Path(
                    output_dir,
                    f"model={key[0]}_forecast_start={str(key[1])}_{plot_name}.png",
                )
            )


def diagnostic_table(
    models: Dict[Tuple[str, dt.date], iup.models.UptakeModel],
    config: Dict[str, Any],
    output_dir,
):
    """select the fitted model using model name and training end date
    and generate selected diagnostics: summary/posterior as parquet"""

    sel_model_dict = select_model_to_diagnose(models, config)

    diagnose_table_names = config["diagnostics"]["table"]

    for key, model in sel_model_dict.items():
        for table_name in diagnose_table_names:
            table_func = getattr(iup.diagnostics, table_name)
            output = table_func(model)

            output.write_parquet(
                Path(
                    output_dir,
                    f"model={key[0]}_forecast_start={str(key[1])}_{table_name}.parquet",
                )
            )


def select_model_to_diagnose(
    models: Dict[Tuple[str, dt.date], iup.models.UptakeModel], config
) -> dict:
    """Select the model to diagnose based on the model name and the training end date"""

    forecast_dates = config["diagnostics"]["forecast_date"]

    if forecast_dates is None:
        sel_keys = [
            (model, date)
            for model, date in models.keys()
            if model in config["diagnostics"]["model"]
        ]
    else:
        assert isinstance(forecast_dates, list)
        assert all(isinstance(x, dt.date) for x in forecast_dates)
        sel_keys = [
            (model, date)
            for model, date in models.keys()
            if model in config["diagnostics"]["model"] and date in forecast_dates
        ]

    return {key: models[key] for key in sel_keys}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--fits", help="fits pickle")
    p.add_argument(
        "--output", help="output status file; other files put in the same directory"
    )
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.fits, "rb") as f:
        models = pickle.load(f)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # write the other plots to the same folder
    diagnostic_plot(models, config, output_dir)
    diagnostic_table(models, config, output_dir)

    # write the status file
    with open(args.output, "w") as f:
        f.write(dt.datetime.now().isoformat())

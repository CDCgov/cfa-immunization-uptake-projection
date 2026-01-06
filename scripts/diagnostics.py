import argparse
import datetime as dt
import pickle
from pathlib import Path

import yaml

import iup
import iup.diagnostics
import iup.models


def diagnostic_plot(
    plot_name: str, fit: iup.models.UptakeModel, output_path: str | Path
):
    plot_func = getattr(iup.diagnostics, plot_name)
    axes = plot_func(fit)
    fig = axes.ravel()[0].figure
    fig.savefig(output_path)


def diagnostic_table(
    table_name: str, fit: iup.models.UptakeModel, output_path: str | Path
):
    table_func = getattr(iup.diagnostics, table_name)
    output = table_func(fit)
    output.write_csv(output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--fits_dir", help="directory with fit pickles", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for key in ["forecast_starts", "models", "tables", "plots"]:
        assert isinstance(config["diagnostics"][key], list), (
            f"config['diagnostics']['{key}'] should be a list"
        )

    for forecast_start in config["diagnostics"]["forecast_starts"]:
        fc_date = dt.date.fromisoformat(forecast_start)

        for model in config["diagnostics"]["models"]:
            with open(Path(args.fits_dir) / f"fit_{fc_date}.pkl", "rb") as f:
                fits = pickle.load(f)

            fit = fits[(model, fc_date)]

            for table in config["diagnostics"]["tables"]:
                diagnostic_table(
                    table_name=table,
                    fit=fit,
                    output_path=Path(
                        args.output_dir,
                        f"model={model}_forecast_start={fc_date}_{table}.csv",
                    ),
                )

            for plot in config["diagnostics"]["plots"]:
                diagnostic_plot(
                    plot_name=plot,
                    fit=fit,
                    output_path=Path(
                        args.output_dir,
                        f"model={model}_forecast_start={fc_date}_{plot}.png",
                    ),
                )

import argparse
import datetime as dt
from typing import Any, List, Type

import arviz as az
import polars as pl
import yaml

import iup
import iup.models


def run_all_forecasts(data, config) -> dict:
    """Run all forecasts

    Returns:
        dictionary of two data frames: forecasts and posterior distributions,
        both organized by model and forecast date
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

    all_forecast = pl.DataFrame()
    all_posterior = []

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
            config["data"]["groups"],
            config["data"]["rollouts"],
        )

        model_posterior = pl.DataFrame()

        for forecast_date in forecast_dates:
            model_output = run_forecast(
                data=augmented_data,
                model_class=model_class,
                seed=config_model["seed"],
                params=config_model["params"],
                mcmc=config_model["mcmc"],
                grouping_factors=config["data"]["groups"],
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                forecast_interval=config["forecast_timeframe"]["interval"],
                season_start_month=config["data"]["season_start_month"],
                season_start_day=config["data"]["season_start_day"],
            )

            forecast = model_output["projections"]

            forecast = forecast.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                model=pl.lit(model_name),
            )
            all_forecast = pl.concat([all_forecast, forecast])

            posterior = model_output["posterior"]
            posterior = posterior.with_columns(
                forecast_start=forecast_date,
                forecast_end=config["forecast_timeframe"]["end"],
                model=pl.lit(model_name),
            )
            model_posterior = pl.concat([model_posterior, posterior])

        all_posterior.append(model_posterior)

    return {"forecasts": all_forecast, "posteriors": all_posterior}


def run_forecast(
    data: iup.UptakeData,
    model_class: Type[iup.models.UptakeModel],
    seed: int,
    params: dict[str, Any],
    mcmc: dict[str, Any],
    grouping_factors: List[str] | None,
    forecast_start: dt.date,
    forecast_end: dt.date,
    forecast_interval: str,
    season_start_month: int,
    season_start_day: int,
) -> dict:
    """
    Run a single model for a single forecast date.
    Return the model posterior and the projections.
    """
    train_data = iup.UptakeData.split_train_test(data, forecast_start, "train")

    # Make an instance of the model, fit it using training data, and make projections
    fit_model = model_class(seed).fit(
        train_data,
        grouping_factors,
        params,
        mcmc,
    )

    # Extract the posterior distribution from the model as a data frame
    posterior = pl.from_pandas(
        az.from_numpyro(fit_model.mcmc).to_dataframe(
            groups="posterior", include_coords=False
        )
    )

    # Rename columns using the actual levels of grouping factors, not numeric codes
    if fit_model.value_to_index is not None:
        group_factors = list(fit_model.value_to_index.keys())
        group_levels = [
            k
            for inner_dict in fit_model.value_to_index.values()
            for k in inner_dict.keys()
        ]
        group_factors_dict = {
            "[" + str(i) + "]": "_" + v.replace(" ", "_")
            for i, v in enumerate(group_factors)
        }
        group_levels_dict = {
            "[" + str(i) + "]": "_" + v.replace(" ", "_")
            for i, v in enumerate(group_levels)
        }
        for k, v in group_factors_dict.items():
            posterior = posterior.rename(
                {
                    col: col.replace(k, v) if "sigs" in col else col
                    for col in posterior.columns
                }
            )
        for k, v in group_levels_dict.items():
            posterior = posterior.rename(
                {
                    col: col.replace(k, v) if "devs" in col else col
                    for col in posterior.columns
                }
            )

    # Get test data, if there is any, to know exact dates for projection
    test_data = iup.UptakeData.split_train_test(data, forecast_start, "test")
    if test_data.height == 0:
        test_data = None

    cumulative_projections = fit_model.predict(
        start_date=forecast_start,
        end_date=forecast_end,
        interval=forecast_interval,
        test_data=test_data,
        groups=grouping_factors,
        season_start_month=season_start_month,
        season_start_day=season_start_day,
    )

    return {"posterior": posterior, "projections": cumulative_projections}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--input", help="input data")
    p.add_argument("--output_forecast", help="output parquet file for forecasts")
    p.add_argument(
        "--output_posterior", help="output parquet file for posterior distributions"
    )
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(pl.scan_parquet(args.input).collect())

    output = run_all_forecasts(input_data, config)

    output["forecasts"].write_parquet(args.output_forecast)

    file_name_parts = args.output_posterior.split(".")
    for i in range(len(output["posteriors"])):
        model_name = output["posteriors"][i]["model"][0]
        file_name = (
            "".join(file_name_parts[:-1])
            + "_"
            + model_name
            + "."
            + "".join(file_name_parts[-1])
        )
        output["posteriors"][i].write_parquet(file_name)

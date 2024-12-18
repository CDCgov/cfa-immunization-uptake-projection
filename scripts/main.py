import argparse

import nisapi
import polars as pl
import yaml

import iup
import iup.eval
import iup.models


def run(config: dict, cache: str):
    forecast_dates = pl.date_range(
        config["timeframe"]["start"],
        config["timeframe"]["end"],
        config["timeframe"]["interval"],
        eager=True,
    )
    models = [getattr(iup.models, model_name) for model_name in config["models"]]
    assert all(issubclass(model, iup.models.UptakeModel) for model in models)
    score_funs = [getattr(iup.eval, score_name) for score_name in config["score_funs"]]

    # Preprocessing -----------------------------------------------------------
    # Get uptake data from the cache
    data = nisapi.get_nis(cache)

    # Prune data to correct rows and columns
    cumulative_data = [
        iup.CumulativeUptakeData(
            data.filter(**x["filters"])
            .collect()
            .select(config["keep"])
            .sort("time_end")
        )
        for x in config["data"].values()
    ]

    # Ensure that the desired grouping factors are found in all data sets
    grouping_factors = config["groups"]
    assert all(g in df.columns for g in grouping_factors for df in cumulative_data)

    # Insert rollout dates into the data
    cumulative_data = [
        iup.CumulativeUptakeData(x.insert_rollout(y["rollout"], grouping_factors))
        for x, y in zip(cumulative_data, config["data"].values())
    ]

    # List of incident data sets from the cumulative data sets
    incident_data = [x.to_incident(grouping_factors) for x in cumulative_data]

    # Forecasts ---------------------------------------------------------------

    for model in models:
        for forecast_date in forecast_dates:
            # Get data available as of the forecast date
            incident_train_data = iup.IncidentUptakeData(
                iup.IncidentUptakeData.split_train_test(
                    incident_data, config["timeframe"]["start"], "train"
                )
            )

            # Fit models using the training data and make projections
            fit_model = model().fit(incident_train_data, grouping_factors)

            cumulative_projections = fit_model.predict(
                config["timeframe"]["start"],
                config["timeframe"]["end"],
                config["timeframe"]["interval"],
                grouping_factors,
            )
            # save these projections somewhere

            incident_projections = cumulative_projections.to_incident(grouping_factors)
            # save these projections somewhere

            # Evaluation / Post-processing --------------------------------------------

            incident_test_data = iup.IncidentUptakeData(
                iup.IncidentUptakeData.split_train_test(
                    incident_data, config["timeframe"]["start"], "test"
                )
            ).filter(pl.col("date") <= config["timeframe"]["end"])

            for score_fun in score_funs:
                score = eval.score(incident_test_data, incident_projections, score_fun)
                print(f"{model=} {forecast_date=} {score_fun=} {score=}")
                # save these scores somewhere


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument(
        "--cache", help="NIS cache directory", default=".cache/nisapi/clean/"
    )
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(config=config, cache=args.cache)

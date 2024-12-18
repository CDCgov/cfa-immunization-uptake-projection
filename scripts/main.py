import argparse
from pathlib import Path

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
    raw_data = nisapi.get_nis(Path(cache))

    for data_set_spec in config["data"]:
        # Prune data to correct rows and columns
        cumulative_data = iup.CumulativeUptakeData(
            raw_data.filter(**data_set_spec["filters"])
            .select(data_set_spec["keep"])
            .sort("time_end")
            .collect()
        )

        # Ensure that the desired grouping factors are found in all data sets
        grouping_factors = config["groups"]
        assert set(cumulative_data.columns).issuperset(grouping_factors)

        # Insert rollout dates into the data
        cumulative_data = iup.CumulativeUptakeData(
            cumulative_data.insert_rollout(data_set_spec["rollout"], grouping_factors)
        )

        # Convert to incident data
        incident_data = cumulative_data.to_incident(grouping_factors)

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

                incident_projections = cumulative_projections.to_incident(
                    grouping_factors
                )
                # save these projections somewhere

                # Evaluation / Post-processing --------------------------------------------

                incident_test_data = iup.IncidentUptakeData(
                    iup.IncidentUptakeData.split_train_test(
                        incident_data, config["timeframe"]["start"], "test"
                    )
                ).filter(pl.col("date") <= config["timeframe"]["end"])

                for score_fun in score_funs:
                    score = eval.score(
                        incident_test_data, incident_projections, score_fun
                    )
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

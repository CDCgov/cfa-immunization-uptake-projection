import argparse

import nisapi
import yaml

import iup
from iup.models import LinearIncidentUptakeModel


def run(config: dict, cache: str):
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

    # Insure that the desired grouping factors are found in all data sets
    grouping_factors = config["groups"]
    assert all(g in df.columns for g in grouping_factors for df in cumulative_data)

    # Insert rollout dates into the data
    cumulative_data = [
        iup.CumulativeUptakeData(x.insert_rollout(y["rollout"], grouping_factors))
        for x, y in zip(cumulative_data, config["data"].values())
    ]

    # List of incident data sets from the cumulative data sets
    incident_data = [x.to_incident(grouping_factors) for x in cumulative_data]

    # Concatenate data sets and split into train and test subsets
    incident_train_data = iup.IncidentUptakeData(
        iup.IncidentUptakeData.split_train_test(
            incident_data, config["timeframe"]["start"], "train"
        )
    )

    # Fit models using the training data and make projections
    incident_model = LinearIncidentUptakeModel().fit(
        incident_train_data, grouping_factors
    )
    cumulative_projections = incident_model.predict(
        config["timeframe"]["start"],
        config["timeframe"]["end"],
        config["timeframe"]["interval"],
        grouping_factors,
    )
    print(cumulative_projections)
    incident_projections = cumulative_projections.to_incident(grouping_factors)
    print(incident_projections)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--cache", help="NIS cache directory")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(config=config, cache=args.cache)

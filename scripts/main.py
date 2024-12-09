import yaml
import iup
from iup.models import LinearIncidentUptakeModel
import argparse


def run(config):
    # List of the cumulative data sets described in the yaml
    cumulative_data = [iup.parse_nis(**x) for x in config["data"].values()]

    # List of grouping factors used in each data set
    grouping_factors = iup.extract_group_names(
        [x["group_cols"] for x in config["data"].values()]
    )

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
    p.add_argument("--config", type=str, default="config.yaml", dest="config")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(config)

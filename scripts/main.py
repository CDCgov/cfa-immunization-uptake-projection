import yaml
import iup
import argparse


def run(config):
    # List of the cumulative data sets described in the yaml
    cumulative_data = [iup.parse_nis(**x) for x in config["data"].values()]

    # List of incident data sets from the cumulative data sets
    incident_data = [x.to_incident() for x in cumulative_data]

    # Concatenate data sets and split into train and test subsets
    # cumulative_train_data = iup.CumulativeUptakeData.split_train_test(
    #    cumulative_data, config["timeframe"]["start"], "train"
    # )
    # cumulative_test_data = iup.CumulativeUptakeData.split_train_test(
    #    cumulative_data, config["timeframe"]["start"], "test"
    # )
    incident_train_data = iup.IncidentUptakeData.split_train_test(
        incident_data, config["timeframe"]["start"], "train"
    )
    # incident_test_data = iup.IncidentUptakeData.split_train_test(
    #    incident_data, config["timeframe"]["start"], "test"
    # )

    # Fit models using the training data and make projections
    incident_model = (
        iup.LinearIncidentUptakeModel()
        .fit(incident_train_data)
        .predict(
            config["timeframe"]["start"],
            config["timeframe"]["end"],
            config["timeframe"]["interval"],
        )
    )
    print(incident_model.cumulative_projection)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml", dest="config")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(config)
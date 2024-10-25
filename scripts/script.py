import yaml
import iup

# Access the YAML config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# List of the cumulative data sets described in the yaml
cumulative_data = [
    iup.get_nis(
        x["path"],
        x["region_col"],
        x["date_col"],
        x["estimate_col"],
        x["rollout"],
        x["filters"],
    )
    for x in config["data"].values()
]

# List of incident data sets from the cumulative data sets
incident_data = [x.to_incident() for x in cumulative_data]

# Concatenate data sets and split into train and test subsets
cumulative_train_data = iup.CumulativeUptakeData.split_train_test(
    cumulative_data, config["timeframe"]["start"], "train"
)
cumulative_test_data = iup.CumulativeUptakeData.split_train_test(
    cumulative_data, config["timeframe"]["start"], "test"
)
incident_train_data = iup.IncidentUptakeData.split_train_test(
    incident_data, config["timeframe"]["start"], "train"
)
incident_test_data = iup.IncidentUptakeData.split_train_test(
    incident_data, config["timeframe"]["start"], "test"
)

# Figure out how to initialize and fit models using the training data
incident_model = iup.LinearIncidentUptakeModel().fit(incident_train_data).predict(
    config["timeframe"]["start"],
    config["timeframe"]["end"],
    config["timeframe"]["interval"],
)

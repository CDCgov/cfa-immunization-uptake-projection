import main
import yaml
import iup
import argparse


def eval(config, model):
    # Read the cumulative data at a given time frame specified in the yaml

    # Convert cumulative data to incidence data

    # Fit the model with LIUM and incidence data

    # Retrospective forecasting: iteratively generate projections given a time frame, using the same model

    # evaluate metrics #
    # mspe #

    # mean bias #

    # absolute error of end-of-season uptake #
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml", dest="config")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = main.run(config)
    eval(config, model)

import argparse
import datetime
from typing import List

import nisapi
import polars as pl
import yaml

import iup


def preprocess(
    raw_data: pl.LazyFrame,
    filters: dict,
    keep: List[str],
    groups: List[str],
    rollout_dates: List[datetime.date],
) -> pl.DataFrame:
    # Prune data to correct rows and columns
    cumulative_data = iup.CumulativeUptakeData(
        raw_data.filter(**filters).select(keep).sort("time_end").collect()
    )

    # Ensure that the desired grouping factors are found in all data sets
    assert set(cumulative_data.columns).issuperset(groups)

    # Insert rollout dates into the data
    cumulative_data = iup.CumulativeUptakeData(
        cumulative_data.insert_rollout(rollout_dates, groups)
    )

    # Convert to incident data
    incident_data = cumulative_data.to_incident(groups)

    return pl.concat(
        [
            cumulative_data.with_columns(estimate_type=pl.lit("cumulative")),
            incident_data.with_columns(estimate_type=pl.lit("incident")),
        ]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument(
        "--cache", help="NIS cache directory", default=".cache/nisapi/clean/"
    )
    # p.add_argument("--cache", help="clean cache directory")
    # comment out the above because an error occurs with 'conflicting --cache' if not

    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    assert len(config["data"]) == 1, "Don't know how to preprocess multiple data sets"

    raw_data = nisapi.get_nis(path=args.cache)

    clean_data = preprocess(
        raw_data,
        filters=config["data"]["data_set_1"]["filters"],
        keep=config["keep"],
        groups=config["groups"],
        rollout_dates=config["data"]["data_set_1"]["rollout"],
    )

    clean_data.write_parquet(args.output)

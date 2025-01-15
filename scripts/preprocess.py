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
    rollouts: List[datetime.date],
) -> iup.CumulativeUptakeData:
    # Prune data to correct rows and columns
    data = raw_data.filter(**filters).select(keep).sort("time_end").collect()

    # Ensure that the desired grouping factors are found in all data sets
    assert set(data.columns).issuperset(groups)

    # Insert rollout dates into the data
    # note the awkward wrapping with the class, because insert_rollouts returns
    # a normal data frame
    return iup.CumulativeUptakeData(
        iup.CumulativeUptakeData(data).insert_rollouts(rollouts, groups)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument(
        "--cache", help="NIS cache directory", default=".cache/nisapi/clean/"
    )
    p.add_argument("--output", help="output parquet file", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    raw_data = nisapi.get_nis(path=args.cache)

    clean_data = preprocess(
        raw_data,
        filters=config["data"]["filters"],
        keep=config["data"]["keep"],
        groups=config["data"]["groups"],
        rollouts=config["data"]["rollouts"],
    )

    clean_data.write_parquet(args.output)

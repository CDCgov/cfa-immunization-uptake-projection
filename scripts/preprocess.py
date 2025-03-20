import argparse
from typing import List

import nisapi
import polars as pl
import scipy.stats as st
import yaml

import iup
import iup.utils


def preprocess(
    raw_data: pl.LazyFrame,
    filters: dict,
    keep: List[str],
    groups: List[str] | None,
    season_start_month: int,
    season_start_day: int,
) -> iup.CumulativeUptakeData:
    data = iup.CumulativeUptakeData(
        raw_data.filter([pl.col(k).is_in(v) for k, v in filters.items()])
        .select(keep)
        .sort("time_end")
        .collect()
        .with_columns(
            season=pl.col("time_end").pipe(
                iup.utils.date_to_season,
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            ),
            uci=pl.col("uci") - pl.col("estimate"),
            lci=pl.col("estimate") - pl.col("lci"),
        )
        .with_columns(sdev=pl.max_horizontal("uci", "lci") / st.norm.ppf(0.975, 0, 1))
        .drop(["lci", "uci"])
        .with_columns(
            sdev=pl.when(pl.col("sdev") < 0.0000005)
            .then(0.0000005)
            .otherwise(pl.col("sdev"))
        )
    )

    if groups is not None:
        assert set(data.columns).issuperset(groups)

    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument(
        "--cache", help="NIS cache directory", default=".cache/nisapi/clean/"
    )
    p.add_argument("--output", help="output parquet file", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_data = nisapi.get_nis(path=args.cache)

    clean_data = preprocess(
        raw_data,
        filters=config["data"]["filters"],
        keep=config["data"]["keep"],
        groups=config["data"]["groups"],
        season_start_month=config["data"]["season_start_month"],
        season_start_day=config["data"]["season_start_day"],
    )

    clean_data.write_parquet(args.output)

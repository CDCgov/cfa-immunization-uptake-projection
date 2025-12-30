import argparse
from datetime import date
from pathlib import Path
from typing import List

import polars as pl
import yaml

import iup
import iup.utils


def preprocess(
    raw_data: pl.LazyFrame,
    groups: List[str] | None,
    season_start_month: int,
    season_start_day: int,
    geographies: List[str] | None,
) -> iup.CumulativeUptakeData:
    # filter for specific geographies
    def geo_filter(df: pl.LazyFrame) -> pl.LazyFrame:
        if geographies is None:
            return df
        else:
            return df.filter(pl.col("geography").is_in(geographies))

    data = iup.CumulativeUptakeData(
        raw_data.rename({"sample_size": "N_tot"})
        .with_columns(
            season=iup.utils.date_to_season(
                pl.col("time_end"),
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            ),
            N_vax=(pl.col("N_tot") * pl.col("estimate")).round(0),
            t=iup.utils.date_to_elapsed(
                pl.col("time_end"),
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            ),
        )
        .filter(
            # drop the nation
            pl.col("geography_type") == pl.lit("admin1"),
            # remove territories
            pl.col("geography")
            .is_in(["Puerto Rico", "U.S. Virgin Islands", "Guam"])
            .not_(),
            # remove data that don't fit nicely into seasons
            pl.col("time_end").is_between(
                date(
                    config["season"]["first_year"],
                    config["season"]["start_month"],
                    config["season"]["start_day"],
                ),
                date(
                    config["season"]["last_year"],
                    config["season"]["end_month"],
                    config["season"]["end_day"],
                ),
            ),
        )
        .pipe(geo_filter)
        .sort("time_end")
        .collect()
    )

    if groups is not None:
        assert set(data.columns).issuperset(groups)

    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="output parquet file", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_data = pl.scan_parquet(args.input)

    assert isinstance(config, dict)
    geographies = config.get("geographies", None)

    clean_data = preprocess(
        raw_data,
        groups=config["groups"],
        season_start_month=config["season"]["start_month"],
        season_start_day=config["season"]["start_day"],
        geographies=geographies,
    )

    if clean_data.height == 0:
        raise RuntimeError("No data after preprocessing")

    # ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_data.write_parquet(args.output)

import argparse
from pathlib import Path
from typing import List

import nisapi
import polars as pl
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
        .rename({"sample_size": "N_tot"})
        .with_columns(
            season=pl.col("time_end").pipe(
                iup.utils.date_to_season,
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            ),
            N_vax=(pl.col("N_tot") * pl.col("estimate")).round(0),
        )
    )

    if groups is not None:
        assert set(data.columns).issuperset(groups)

    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--output", help="output directory", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_data = nisapi.get_nis()

    clean_data = preprocess(
        raw_data,
        filters=config["data"]["filters"],
        keep=config["data"]["keep"],
        groups=config["data"]["groups"],
        season_start_month=config["data"]["season_start_month"],
        season_start_day=config["data"]["season_start_day"],
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    clean_data.write_parquet(Path(args.output, "nis_data.parquet"))

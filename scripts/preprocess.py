import argparse
from pathlib import Path
from typing import List, Optional

import polars as pl
import yaml

from iup import to_season


def preprocess(
    raw_data: pl.DataFrame,
    start_year: int,
    end_year: int,
    season_start_month: int,
    season_start_day: int,
    season_end_month: int,
    season_end_day: int,
    geographies: Optional[List[str] | None],
    date_col: str = "time_end",
) -> pl.DataFrame:
    """
    Preprocess the raw data (Filter the raw data with certain states and seasons, add season column).

    Args:
        raw_data: Raw Lazy data frame
        season_start_year: The year of the first season to include in the data.
        season_start_month: The month of the first season to include in the data.
        season_start_day: The day of the first season to include in the data.
        season_end_year: The year of the last season to include in the data.
        season_end_month: The month of the last season to include in the data.
        season_end_day: The day of the last season to include in the data.
        geographies: List of geographies to include in the data. If None, include all geographies.

    Returns:
        Preprocessed data frame ready for downstreamed process for two models.

    """

    def geo_filter(df: pl.DataFrame) -> pl.DataFrame:
        if geographies is None:
            return df
        else:
            return df.filter(pl.col("geography").is_in(geographies))

    return (
        raw_data.filter(
            pl.col("geography_type") == pl.lit("admin1"),
            pl.col("geography")
            .is_in(["Puerto Rico", "U.S. Virgin Islands", "Guam"])
            .not_(),
        )
        .with_columns(
            season=to_season(
                pl.col(date_col),
                season_start_month=season_start_month,
                season_start_day=season_start_day,
                season_end_month=season_end_month,
                season_end_day=season_end_day,
            )
        )
        .filter(
            # drop dates before or after the outermost season
            pl.col(date_col).dt.year().is_between(start_year, end_year),
            # drop out-of-season dates between seasons
            pl.col("season").is_null().not_(),
        )
        .pipe(geo_filter)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="output parquet file", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_data = pl.read_parquet(args.input)

    assert isinstance(config, dict)
    geographies = config.get("geographies", None)

    clean_data = preprocess(
        raw_data,
        start_year=config["season"]["start_year"],
        end_year=config["season"]["end_year"],
        season_start_month=config["season"]["start_month"],
        season_start_day=config["season"]["start_day"],
        season_end_month=config["season"]["end_month"],
        season_end_day=config["season"]["end_day"],
        geographies=geographies,
    )

    if clean_data.height == 0:
        raise RuntimeError("No data after preprocessing")

    # ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_data.write_parquet(args.output)

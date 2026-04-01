import argparse
from pathlib import Path
from typing import List, Optional

import polars as pl
import yaml

from iup import CumulativeCoverageData
from iup.utils import date_to_season


def season_filter(
    df: pl.LazyFrame,
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    col_name="time_end",
) -> pl.LazyFrame:
    """Filter a data frame for dates that are before the season end or after the season start.

    Args:
        df: Data frame to filter.
        start_year: Start year of the first season.
        start_month: First month of the season.
        start_day: First day of the season.
        end_year: End year of the last season.
        end_month: Last month of the season.
        end_day: Last day of the season.
        col_name: Name of the column containing dates. Defaults to "time_end".

    Returns:
        Filtered data frame (assumes summer months are "out of season").
    """
    assert (end_month, end_day) < (start_month, start_day), (
        "Only summer-ending seasons are supported"
    )

    col = pl.col(col_name)
    year = col.dt.year()
    month = col.dt.month()
    day = col.dt.day()

    return (
        df.filter(
            (year > start_year)
            | ((year == start_year) & (month >= start_month) & (day >= start_day))
        ).filter(
            (year < end_year)
            | ((year == end_year) & (month <= end_month) & (day <= end_day))
        )  # remove partial season
    )


def preprocess(
    raw_data: pl.LazyFrame,
    season_start_year: int,
    season_start_month: int,
    season_start_day: int,
    season_end_year: int,
    season_end_month: int,
    season_end_day: int,
    geographies: Optional[List[str] | None],
) -> CumulativeCoverageData:
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

    def geo_filter(df: pl.LazyFrame) -> pl.LazyFrame:
        if geographies is None:
            return df
        else:
            return df.filter(pl.col("geography").is_in(geographies))

    data = (
        raw_data.filter(
            pl.col("geography_type") == pl.lit("admin1"),
            pl.col("geography")
            .is_in(["Puerto Rico", "U.S. Virgin Islands", "Guam"])
            .not_(),
        )
        .with_columns(
            season=date_to_season(
                pl.col("time_end"),
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            )
        )
        .pipe(
            season_filter,
            start_year=season_start_year,
            start_month=season_start_month,
            start_day=season_start_day,
            end_year=season_end_year,
            end_month=season_end_month,
            end_day=season_end_day,
        )
        .pipe(geo_filter)
        .with_columns(
            t=days_in_season(
                pl.col("time_end"),
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            )
        )
        .collect()
    )

    return CumulativeCoverageData(data)


def days_in_season(
    date_col: pl.Expr, season_start_month: int, season_start_day: int
) -> pl.Expr:
    """Extract a time elapsed column from a date column, as polars expressions.

    Args:
        date_col: Column of dates.
        season_start_month: First month of the overwinter disease season.
        season_start_day: First day of the first month of the overwinter disease season.

    Returns:
        number of days elapsed since the first date
    """
    # for every date, figure out the season breakpoint in that year
    season_start = pl.date(date_col.dt.year(), season_start_month, season_start_day)

    # for dates before the season breakpoint in year, subtract a year
    year = date_col.dt.year()
    season_start_year = pl.when(date_col < season_start).then(year - 1).otherwise(year)

    # rewrite the season breakpoints to that immediately before each date
    season_start = pl.date(season_start_year, season_start_month, season_start_day)

    # return the number of days from season start to each date
    return (date_col - season_start).dt.total_days()


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
        season_start_year=config["season"]["start_year"],
        season_start_month=config["season"]["start_month"],
        season_start_day=config["season"]["start_day"],
        season_end_year=config["season"]["end_year"],
        season_end_month=config["season"]["end_month"],
        season_end_day=config["season"]["end_day"],
        geographies=geographies,
    )

    if clean_data.height == 0:
        raise RuntimeError("No data after preprocessing")

    # ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_data.write_parquet(args.output)

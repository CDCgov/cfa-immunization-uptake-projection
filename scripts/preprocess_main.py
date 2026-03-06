import argparse
from datetime import date
from pathlib import Path
from typing import List

import polars as pl
import yaml

import iup
import iup.utils


from pathlib import Path
from typing import Tuple

import altair as alt
import forestci as fci
import numpy as np
import polars as pl
from plot_data import month_order
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

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
    raw_data,
    season_start_year,
    season_start_month,
    season_start_day,
    season_end_year,
    season_end_month,
    season_end_day,
    geographies,
):
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
        .collect()
    )

    return data


raw_data = pl.scan_parquet("data/raw.parquet")

clean_data = preprocess(raw_data, 2009, 7, 1, 2022, 4, 1, ["Alaska"])


def preprocess_lpl(main_data, groups, season_start_month, season_start_day):
    main_data = iup.CumulativeCoverageData(
        main_data.rename({"sample_size": "N_tot"})
    ).with_columns(
        N_vax=(pl.col("N_tot") * pl.col("estimate")).round(0),
        season_geo=pl.concat_str(["season", "geography"], separator="_"),
        t=iup.utils.date_to_elapsed(
            pl.col("time_end"),
            season_start_month=season_start_month,
            season_start_day=season_start_day,
        ),
    )

    if groups is not None:
        assert set(main_data.columns).issuperset(groups)

    return main_data


lpl_data = preprocess_lpl(clean_data, ["season", "geography", "season_geo"], 7, 1)
# pl.Config.set_tbl_cols(-1)
# print(lpl_data)

# def preprocess_rf(main_data, )

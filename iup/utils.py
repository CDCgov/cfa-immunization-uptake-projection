from typing import List

import numpy as np
import polars as pl


def date_to_season(
    date: pl.Expr, season_start_month: int, season_start_day: int = 1
) -> pl.Expr:
    """
    Extract the overwinter disease season from a date

    Parameters
    date: pl.Expr
        dates in an uptake data frame
    season_start_month: int
        first month of the overwinter disease season
    season_start_day: int
        first day of the first month of the overwinter disease season

    Returns
    pl.Expr
        seasons for each date

    Details
    Dates in year Y before the season start (e.g., Sep 1) are in the second part of
    the season (i.e., in season Y-1/Y). Dates in year Y after the season start are in
    season Y/Y+1. E.g., 2023-10-07 and 2024-04-18 are both in "2023/2024"
    """

    # for every date, figure out the season breakpoint in that year
    season_start = pl.date(date.dt.year(), season_start_month, season_start_day)

    # what is the first year in the two-year season indicator?
    date_year = date.dt.year()
    year1 = pl.when(date < season_start).then(date_year - 1).otherwise(date_year)

    year2 = year1 + 1
    return pl.format("{}/{}", year1, year2)


def date_to_elapsed(
    date_col: pl.Expr, season_start_month=0, season_start_day=0
) -> pl.Expr:
    """
    Extract a time elapsed column from a date column, as polars expressions.

    Parameters
    date_col: pl.Expr
        column of dates
    season_start_month: int
        first month of the overwinter disease season
    season_start_day: int
        first day of the first month of the overwinter disease season

    Returns
    pl.Expr
        column of the number of days elapsed since the first date

    Details
    Date column should be chronologically sorted in advance.
    Time difference is always in days.
    If a season start month and day is provided,
    time elapsed is calculated since the season start.
    Otherwise, time elapsed is calculated since the first report date in a season.
    This ought to be called .over(season)
    """

    if season_start_month == 0 and season_start_day == 0:
        return (date_col - date_col.first()).dt.total_days()

    else:
        # for every date, figure out the season breakpoint in that year
        season_start = pl.date(date_col.dt.year(), season_start_month, season_start_day)

        # for dates before the season breakpoint in year, subtract a year
        year = date_col.dt.year()
        season_start_year = (
            pl.when(date_col < season_start).then(year - 1).otherwise(year)
        )

        # rewrite the season breakpoints to that immediately before each date
        season_start = pl.date(season_start_year, season_start_month, season_start_day)

        # return the number of days from season start to each date
        return (date_col - season_start).dt.total_days()


def map_value_to_index(groups: pl.DataFrame) -> dict:
    """
    Choose a numeric index for each level of each grouping factor in a data frame.

    Parameters
    groups: pl.DataFrame
        levels of grouping factors (cols) for multiple data points (rows)

    Returns
    dict
        {grouping_factor => {value => integer_index}}
    """
    return {
        col: {
            value: i
            for i, value in enumerate(
                groups.select(pl.col(col).unique().sort()).to_series()
            )
        }
        for col in groups.columns
    }


def value_to_index(
    groups: pl.DataFrame, mapping: dict, num_group_levels: List[int,]
) -> np.ndarray:
    """
    Replace each level of each grouping factor in a data frame, using a pre-determined mapping.

    Parameters
    groups: pl.DataFrame
        levels of grouping factors (cols) for multiple data points (rows)
    mapping: dict
        mapping of each level of each grouping factor to a numeric code
    num_group_levels: bool
        total number of levels for each grouping factor

    Returns
    np.ndarray
        array of group levels but with numeric codes instead of level names

    Details
    Numeric codes will be used only once across grouping factors.
    The keys of mapping must match the column names of groups.
    """
    assert set(mapping.keys()) == set(groups.columns), (
        "Keys of mapping do not match grouping factor names."
    )

    for col_name in groups.columns:
        if missing_values := set(
            groups.select(pl.col(col_name).unique()).to_series()
        ) - set(mapping[col_name].keys()):
            raise RuntimeError(f"Missing indices for values: {missing_values}")

        groups = groups.with_columns(
            pl.col(col_name).replace_strict(mapping[col_name]).cast(pl.UInt8)
        )

    array = groups.to_numpy() + np.cumsum([0] + num_group_levels[:-1])

    return array


def count_unique_values(df: pl.DataFrame | None) -> List[int,]:
    """
    Count unique values in each column of a data frame

    Parameters
    df: pl.DataFrame

    Returns
    List[int,]
        Number of unique values in each column of the data frame
    """
    if df is None:
        return [0]
    else:
        return [df.n_unique(subset=col) for col in df.columns]

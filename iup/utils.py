from typing import Any, List

import numpy as np
import polars as pl


def date_to_season(
    date: pl.Expr, season_start_month: int, season_start_day: int = 1
) -> pl.Expr:
    """Extract the overwinter disease season from a date.

    Dates in year Y before the season start (e.g., Sep 1) are in the second part of
    the season (i.e., in season Y-1/Y). Dates in year Y after the season start are in
    season Y/Y+1. E.g., 2023-10-07 and 2024-04-18 are both in "2023/2024".

    Args:
        date: Dates in an coverage data frame.
        season_start_month: First month of the overwinter disease season.
        season_start_day: First day of the first month of the overwinter disease season.

    Returns:
        Seasons for each date.
    """

    # for every date, figure out the season breakpoint in that year
    season_start = pl.date(date.dt.year(), season_start_month, season_start_day)

    # what is the first year in the two-year season indicator?
    date_year = date.dt.year()
    year1 = pl.when(date < season_start).then(date_year - 1).otherwise(date_year)

    year2 = year1 + 1
    return pl.format("{}/{}", year1, year2)


def date_to_elapsed(
    date_col: pl.Expr, season_start_month: int, season_start_day: int
) -> pl.Expr:
    """Extract a time elapsed column from a date column, as polars expressions.

    Args:
        date_col: Column of dates.
        season_start_month: First month of the overwinter disease season.
        season_start_day: First day of the first month of the overwinter disease season.

    Returns:
        Column of the number of days elapsed since the first date.

    Note:
        Dates should be chronologically sorted in advance.
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


def map_value_to_index(groups: pl.DataFrame) -> dict[str, dict[Any, int]]:
    """Choose a numeric index for each level of each grouping factor in a data frame.

    Args:
        groups: Levels of grouping factors (cols) for multiple data points (rows).

    Returns:
        dictionary of dictionaries {grouping_factor => {value => integer_index}}
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
    """Replace each level of each grouping factor in a data frame, using a pre-determined mapping.

    Numeric codes will be used only once across grouping factors.
    The keys of mapping must match the column names of groups.

    Args:
        groups: Levels of grouping factors (cols) for multiple data points (rows).
        mapping: Mapping of each level of each grouping factor to a numeric code.
        num_group_levels: Total number of levels for each grouping factor.

    Returns:
        Array of group levels but with numeric codes instead of level names.
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
    """Count unique values in each column of a data frame.

    Args:
        df: Data frame to count unique values in.

    Returns:
        Number of unique values in each column of the data frame.
    """
    if df is None:
        return [0]
    else:
        return [df.n_unique(subset=col) for col in df.columns]

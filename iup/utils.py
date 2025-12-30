from typing import Any, Tuple

import numpy as np
import polars as pl


def date_to_season(
    date: pl.Expr, season_start_month: int = 9, season_start_day: int = 1
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


def map_level_to_index(groups: pl.DataFrame) -> dict[Tuple[str, Any], int]:
    """
    Choose a numeric index for each level (i.e., factor and value) in a data frame.
    There is a single set of indices for all factors (e.g., index 0 refer to a
    particular factor *and* level, like season 2019/2020).

    Parameters
    groups: pl.DataFrame
        levels of grouping factors (cols) for multiple data points (rows)

    Returns
    dict
        { (factor, level) => index }, e.g., { ("season", "2019/2020") => 0 }
    """
    return {
        x: i
        for i, x in enumerate(
            sorted(
                set(
                    (factor, level)
                    for factor in groups.columns
                    for level in groups[factor]
                )
            )
        )
    }


def get_design_matrices(
    groups: pl.DataFrame, level_to_index: dict[Tuple[str, Any], int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create design matrices

    Args:
        groups: each column is a factor; each row is an observation; values are
            the levels of each factor for that observation
        level_to_index: see map_level_to_index()

    Returns: data-level matrix and level-factor matrix. The data-level matrix
        has row i column j = 1 if observation i is associated with level j
        (where the index of the level if from `level_to_index`); otherwise 0.
        The level-factor matrix has row i column j = 1 if level i is associated
        with factor j (where the order of the factors is from the order of
        columns in `groups`); otherwise 0.
    """
    data_level_matrix = np.zeros((groups.height, len(level_to_index)))
    for i, row in enumerate(groups.rows(named=True)):
        for factor, level in row.items():
            j = level_to_index[(factor, level)]
            data_level_matrix[i, j] = 1

    factors = groups.columns
    level_factor_matrix = np.zeros((len(level_to_index), groups.width))
    for (factor, level), i in level_to_index.items():
        j = factors.index(factor)
        level_factor_matrix[i, j] = 1

    return (data_level_matrix, level_factor_matrix)

from typing import List

import numpy as np
import polars as pl


def standardize(x, mn=None, sd=None):
    """
    Standardize: subtract mean and divide by standard deviation.

    Parameters
    x: pl.Expr | np.ndarray
        the numbers to standardize
    mn: float
        the term to subtract, if not the mean of x
    sd: float
        the term to divide by, if not the standard deviation of x

    Returns
    pl.Expr | float
        the standardized numbers

    Details
    If the standard deviation is 0, all standardized values are 0.0.
    """
    assert mn is not None or sd is not None and type(x) is np.ndarray, (
        "Calculating mean and std from a numpy array will not ignore NaNs!"
    )

    loc = mn if mn is not None else x.mean()
    scale = sd if sd is not None else x.std(ddof=0)

    if scale == 0.0:
        return x * 0.0
    else:
        return (x - loc) / scale


def unstandardize(x, mn, sd):
    """
    Unstandardize: add standard deviation and multiply by mean.

    Parameters
    x: pl.Expr
        the numbers to unstandardize
    mn: float64
        the term to add
    sd: float64
        the term to multiply by

    Returns
    pl.Expr
        the unstandardized numbers
    """
    return x * sd + mn


def extract_standards(data: pl.DataFrame, var_cols: tuple) -> dict:
    """
    Extract means and standard deviations from data frame columns.

    Parameters
    data: pl.DataFrame
        data frame with some columns to be standardized
    var_cols: (str,)
        column names of variables to be standardized

    Returns
    dict
        means and standard deviations for each variable column

    Details
    Keys are the variable names, and values are themselves
    dictionaries of mean and standard deviation.
    """
    standards = {
        var: {"mean": data[var].mean(), "std": data[var].std()} for var in var_cols
    }

    return standards


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


def date_to_interval(date_col: pl.Expr) -> pl.Expr:
    """
    Extract a time interval column from a date column, as polars expressions.

    Parameters
    date_col: pl.Expr
        column of dates

    Returns
    pl.Expr
        column of the number of days between each date and the previous

    Details
    Date column should be chronologically sorted in advance.
    Time difference is always in days.
    This should be called .over(season)
    """
    return date_col.diff().dt.total_days()


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
        Dictionary of dictionaries: for each grouping factor, a dictionary mapping levels to numeric codes
    """
    mapping = {}
    for i in range(groups.shape[1]):
        col_name = groups.columns[i]
        unique_values = groups.select(col_name).unique().to_series().to_list()
        mapping[col_name] = {v: j for j, v in enumerate(unique_values)}

    return mapping


def value_to_index(groups: pl.DataFrame, mapping: dict, unique=True) -> np.ndarray:
    """
    Replace each level of each grouping factor in a data frame, using a pre-determined mapping.

    Parameters
    groups: pl.DataFrame
        levels of grouping factors (cols) for multiple data points (rows)
    mapping: dict
        mapping of each level of each grouping factor to a numeric code
    unique: bool
        whether numeric codes should be unique across grouping factors

    Returns
    np.ndarray
        array of group levels but with numeric codes instead of level names

    Details
    If unique is False, numeric codes will be reused across grouping factors.
    If unique is True (default), numeric codes will be used only once across grouping factors.
    The keys of mapping must match the column names of groups.
    """
    assert set(mapping.keys()) == set(groups.columns), (
        "Keys of mapping do not match grouping factor names."
    )

    for i in range(groups.shape[1]):
        col_name = groups.columns[i]
        groups = groups.with_columns(
            pl.col(col_name).replace(mapping[col_name]).cast(pl.UInt8).alias(col_name)
        )

    array = groups.to_numpy()

    if unique:
        unique_counts = count_unique_values(array)
        array = array + np.cumsum(np.array([0] + unique_counts[:-1]))

    return array


def count_unique_values(array: np.ndarray) -> List[int,]:
    """
    Count unique values in each column of an array

    Parameters
    array: np.ndarray

    Returns
    List[int,]
        Number of unique values in each column of the array
    """
    return [len(np.unique(array[:, i])) for i in range(array.shape[1])]

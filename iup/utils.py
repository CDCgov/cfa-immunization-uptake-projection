import polars as pl


def standardize(x, mn=None, sd=None):
    """
    Standardize: subtract mean and divide by standard deviation.

    Parameters
    x: pl.Expr | float64
        the numbers to standardize
    mn: float64
        the term to subtract, if not the mean of x
    sd: float64
        the term to divide by, if not the standard deviation of x

    Returns
    pl.Expr | float
        the standardized numbers

    Details
    If the standard deviation is 0, all standardized values are 0.0.
    """
    if type(x) is pl.Expr:
        if mn is not None:
            return (x - mn) / sd
        else:
            return (
                pl.when(x.drop_nulls().n_unique() == 1)
                .then(0.0)
                .otherwise((x - x.mean()) / x.std())
            )
    else:
        if mn is not None:
            return (x - mn) / sd
        else:
            return (x - x.mean()) / x.std()


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

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

import calendar
import datetime
from typing import List

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


def month_order(season_start_month: int) -> List[str]:
    return [
        calendar.month_abbr[i]
        for i in list(range(season_start_month, 12 + 1))
        + list(range(1, season_start_month))
    ]


def index_to_date(start_date: datetime.date, months: int) -> datetime.date:
    total_months = start_date.year * 12 + (start_date.month - 1) + months
    target_year, target_month = divmod(total_months, 12)
    target_month += 1  # convert divisor remainder to month-index

    last_day = (
        (
            datetime.date(target_year, target_month % 12 + 1, 1)
            - datetime.timedelta(days=1)
        ).day
        if target_month < 12
        else 31
    )
    target_day = min(start_date.day, last_day)

    return datetime.date(target_year, target_month, target_day)

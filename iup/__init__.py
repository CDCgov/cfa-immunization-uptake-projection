import polars as pl


def to_season(
    date: pl.Expr,
    season_start_month: int,
    season_end_month: int,
    season_start_day: int = 1,
    season_end_day: int = 1,
) -> pl.Expr:
    """
    Identify the overwinter season from a date.

    Every year, there is a season end (e.g., May 1) and a season start (e.g., Sep 1).
    Dates before the season end are associated with the prior season (e.g., Feb 1, 2020
    belongs to 2019/2020 season). Dates after the season start are associated with the
    next season (e.g., Oct 1, 2020 belongs to 2020/2021). Dates between the season end
    and season start are not in any season (e.g., June 1).

    Args:
        date: dates
        season_start_month: first month
        season_end_month: last month
        season_start_day: first day
        season_end_day: last day

    Returns:
        season like "2020/2021"
    """
    assert (season_start_month, season_start_day) > (
        season_end_month,
        season_end_day,
    ), "Only overwinter seasons are supported"

    # year of this date
    y = date.dt.year()
    # start and end dates of seasons in this year
    end = pl.date(y, season_end_month, season_end_day)
    start = pl.date(y, season_start_month, season_start_day)

    # first year of the two-year season
    sy1 = pl.when(date <= end).then(y - 1).when(date >= start).then(y).otherwise(None)

    return pl.when(sy1.is_null()).then(None).otherwise(pl.format("{}/{}", sy1, sy1 + 1))


def eos_abs_diff(
    obs: pl.DataFrame, pred: pl.DataFrame, grouping_factors: list[str]
) -> pl.DataFrame:
    """Calculate the absolute difference between observed data and prediction for the last date in a season.

    Args:
        obs: Observed data.
        pred: Predicted data.
        grouping_factors: Grouping factor column names (must include 'season').

    Returns:
        Data frame with absolute difference scores for end-of-season dates.
    """
    assert "season" in grouping_factors

    return (
        pred.filter(pl.col("quantile") == 0.5)
        .join(
            obs.filter(
                (pl.col("time_end") == pl.col("time_end").max()).over(grouping_factors)
            ),
            on=["time_end"] + grouping_factors,
            how="right",
        )
        .rename({"estimate_right": "obs", "estimate": "pred"})
        .with_columns(
            score_value=(pl.col("pred") - pl.col("obs")).abs(),
            score_fun=pl.lit("eos_abs_diff"),
        )
        .select(
            grouping_factors + ["model", "forecast_date", "score_fun", "score_value"]
        )
        .drop_nulls()
    )

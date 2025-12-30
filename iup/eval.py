from typing import List

import polars as pl


def mspe(
    obs: pl.DataFrame, pred: pl.DataFrame, grouping_factors: List[str]
) -> pl.DataFrame:
    return (
        pred.group_by(["model", "time_end", "forecast_start"] + grouping_factors)
        .agg(pred_median=pl.col("estimate").median())
        .join(obs, on=["time_end"] + grouping_factors, how="right")
        .with_columns(score_value=(pl.col("estimate") - pl.col("pred_median")) ** 2)
        .group_by(["model", "forecast_start"] + grouping_factors)
        .agg(pl.col("score_value").mean())
        .with_columns(score_fun=pl.lit("mspe"))
    )


def eos_abs_diff(
    obs: pl.DataFrame,
    pred: pl.DataFrame,
    grouping_factors: List[str],
) -> pl.DataFrame:
    """
    Generate a function that calculates the absolute difference between
    observed data and prediction for the last date in a season

    Arguments:
    selected_date: a datetime date object to specify which date to do the calculation
    date_col: a polars column expression used to select the date

    Return:
        A function that takes two polars column expressions to do the calculation.
    """
    assert "season" in grouping_factors

    median_pred = pred.group_by(
        ["model", "time_end", "forecast_start"] + grouping_factors
    ).agg(pred_median=pl.col("estimate").median())

    return (
        obs.filter(
            (pl.col("time_end") == pl.col("time_end").max()).over(grouping_factors)
        )
        .join(median_pred, on=["time_end"] + grouping_factors, how="left")
        .with_columns(
            score_value=(pl.col("estimate") - pl.col("pred_median")).abs(),
            score_fun=pl.lit("eos_abs_diff"),
        )
    )

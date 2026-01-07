from typing import List

import polars as pl

SCORE_COLS = ["model", "forecast_date", "score_fun", "score_value"]


def _ensure_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    if isinstance(df, pl.LazyFrame):
        return df
    elif isinstance(df, pl.DataFrame):
        return df.lazy()
    else:
        raise ValueError(f"Object of class {type(df)} cannot be LazyFrame")


def mspe(
    obs: pl.DataFrame | pl.LazyFrame,
    pred: pl.DataFrame | pl.LazyFrame,
    grouping_factors: List[str],
) -> pl.DataFrame:
    """Mean square prediction error

    Args:
        obs: Data frame with columns `time_end`, `estimate`, and the grouping factors
        pred: Data frame columns `model`, `time_end`, `forecast_date`, `estimate`, and
            the grouping factors.
        grouping_factors: Grouping factor column names.

    Returns:
        Data frame with scores.
    """
    obs = _ensure_lazy(obs)
    pred = _ensure_lazy(pred)

    return (
        pred.group_by(["model", "time_end", "forecast_date"] + grouping_factors)
        .agg(pred_median=pl.col("estimate").median())
        .join(obs, on=["time_end"] + grouping_factors, how="right")
        .with_columns(score_value=(pl.col("estimate") - pl.col("pred_median")) ** 2)
        .group_by(["model", "forecast_date"] + grouping_factors)
        .agg(pl.col("score_value").mean())
        .with_columns(score_fun=pl.lit("mspe"))
        .select(grouping_factors + SCORE_COLS)
        .collect()
    )


def eos_abs_diff(
    obs: pl.DataFrame | pl.LazyFrame,
    pred: pl.DataFrame | pl.LazyFrame,
    grouping_factors: List[str],
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
    obs = _ensure_lazy(obs)
    pred = _ensure_lazy(pred)

    median_pred = pred.group_by(
        ["model", "time_end", "forecast_date"] + grouping_factors
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
        .select(grouping_factors + SCORE_COLS)
        .collect()
    )

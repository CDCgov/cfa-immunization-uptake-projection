from typing import Callable

import polars as pl

from iup import IncidentUptakeData, PointForecast, SampleForecast


###### evaluation metrics #####
def check_date_match(data: IncidentUptakeData, pred: PointForecast):
    """
    Check the dates between data and pred.
    Dates must be 1-on-1 equal and no duplicate.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    Error if conditions fail to meet.

    """
    # sort data and pred by date #
    data = IncidentUptakeData(data.sort("time_end"))
    pred = PointForecast(pred.sort("time_end"))

    # 1. Dates must be 1-on-1 equal
    (data["time_end"] == pred["time_end"]).all()

    # 2. There should not be any duplicated date in either data or prediction.
    assert not (any(data["time_end"].is_duplicated())), (
        "Duplicated dates are found in data and prediction."
    )


def point_score(
    data: IncidentUptakeData,
    pred: PointForecast,
    score_fun: Callable[[pl.Expr, pl.Expr], pl.Expr],
) -> pl.DataFrame:
    """
    Calculate score between observed data and forecast.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast
    score_fun:
        Scoring function. Takes observed and true values.

    Return
    pl.DataFrame with one row: forecast start date, forecast end date, and score

    """
    # validate inputs
    assert isinstance(data, IncidentUptakeData)
    assert isinstance(pred, PointForecast)
    check_date_match(data, pred)

    return (
        data.join(pred, on="time_end", how="inner", validate="1:1")
        .rename({"estimate": "data", "estimate_right": "pred"})
        .select(
            forecast_start=pl.col("time_end").min(),
            forecast_end=pl.col("time_end").max(),
            score=score_fun(pl.col("data"), pl.col("pred")),
        )
    )


def sample_to_quantile_score(
    data: IncidentUptakeData, pred: SampleForecast, quantile: float, score_fun: Callable
):
    """
    Calculate the metrics at quantiles of sample distributions.
    """

    # 1. Convert sample forecast pred into quantiles
    summary_pred = pred.group_by("time_end").agg(pl.col("estimate").quantile(quantile))

    # 2. Calculate the score for each quantile
    return (
        data.join(summary_pred, on="time_end", how="inner", validate="1:1")
        .rename({"estimate": "data", "estimate_right": "pred"})
        .select(
            forecast_start=pl.col("time_end").min(),
            forecast_end=pl.col("time_end").max(),
            score=score_fun(pl.col("data"), pl.col("pred")),
        )
    )


def mspe(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return ((x - y) ** 2).mean()


def abs_diff(df, selected_date, date):
    def abs_diff_date_fun(x, y):
        return df.filter(date == selected_date).with_columns(abs(x - y).alias("score"))

    return abs_diff_date_fun

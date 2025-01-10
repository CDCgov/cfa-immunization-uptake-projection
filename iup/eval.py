from typing import Callable

import polars as pl

from iup import IncidentUptakeData, PointForecast


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


def score(
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


def mspe(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return ((x - y) ** 2).mean()


def mean_bias(pred: pl.Expr, data: pl.Expr) -> pl.Expr:
    """
    Note the bias here is not the classical bias calculated from the posterior distribution.
    The bias here is defined as: at time t,
    bias = -1 if pred_t < data_t; bias = 0 if pred_t == data_t; bias = 1 if pred_t > bias_t

    mean_bias = sum of the bias across time/length of data
    """

    return (pred - data).sign().mean()


def eos_abe(data: pl.Expr, pred: pl.Expr) -> pl.Expr:
    """
    Calculate the absolute error of the total uptake at the end of season between data and prediction
    relative to data.
    """
    cum_data = data.cum_sum().tail(1)
    cum_pred = pred.cum_sum().tail(1)
    return abs(cum_data - cum_pred) / cum_data

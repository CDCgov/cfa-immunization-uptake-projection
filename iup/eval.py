import datetime as dt
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


def mspe(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Calculate Mean Squared Prediction Error with polars column expression
    ---------------------
    Arguments:
    x: either observed data or predictions
    y: either observed data or predictions
    Return:
        Mean Squared Prediction Error as a polars column expression

    """
    return ((x - y) ** 2).mean()


def abs_diff(
    selected_date: dt.date, date_col: pl.Expr
) -> Callable[[pl.Expr, pl.Expr], pl.Expr]:
    """
    Generate a function that calculates the absolute difference between
    observed data and prediction on a certain date.
    ----------------------
    Arguments:
    selected_date: a datetime date object to specify which date to do the calculation
    date_col: a polars column expression used to select the date

    Return:
        A function that takes two polars column expressions to do the calculation.
    """

    lit_date = pl.lit(selected_date)

    def f(x: pl.Expr, y: pl.Expr) -> pl.Expr:
        """
        Calculate the absolute difference between two polars column expressions
        -----------------------
        Arguments:
        x: either observed data or predictions
        y: either observed data or predictions

        Return:
        A polars column expression that returns the absolute difference at the certain date, otherwise None
        """
        return pl.when(date_col == lit_date).then((x - y).abs()).otherwise(None)

    return f

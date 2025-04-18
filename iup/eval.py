import datetime as dt
from typing import Callable, Dict

import polars as pl

from iup import IncidentUptakeData, QuantileForecast


###### evaluation metrics #####
def check_date_match(data: IncidentUptakeData, pred: QuantileForecast):
    """
    Check the dates between data and pred.
    Dates must be 1-on-1 equal and no duplicate.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Can be QuantileForecast or PointForecast

    Return
    Error if conditions fail to meet.

    """
    # sort data and pred by date #
    data = IncidentUptakeData(data.sort("time_end"))
    pred = QuantileForecast(pred.sort("time_end"))

    # 1. Dates must be 1-on-1 equal
    (data["time_end"] == pred["time_end"]).all()

    # 2. There should not be any duplicated date in either data or prediction.
    assert not (any(data["time_end"].is_duplicated())), (
        "Duplicated dates are found in data and prediction."
    )


def summarize_score(
    data: IncidentUptakeData, pred: QuantileForecast, score_funs: Dict[str, Callable]
) -> pl.DataFrame:
    """
    Calculate score between observed data and forecast.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Can be QuantileForecast or PointForecast
    score_funs:
        A dictionary of scoring functions. The key is the name of the score, and the value
        is the scoring function.

    Return
    A pl.DataFrame of scores with information including score name and score values, grouped by quantile, forecast

    """
    assert pred.shape[0] == data.shape[0], (
        "The forecast and the test data do not have the same number of dates."
    )

    assert isinstance(data, IncidentUptakeData)
    assert isinstance(pred, QuantileForecast)
    check_date_match(data, pred)

    joined_df = data.join(pred, on="time_end", how="inner", validate="1:1").rename(
        {"estimate": "data", "estimate_right": "pred"}
    )

    all_scores = pl.DataFrame()
    for score_name in score_funs:
        score = joined_df.select(
            quantile=pl.col("quantile").unique().first(),
            forecast_start=pl.col("time_end").min(),
            forecast_end=pl.col("time_end").max(),
            score_name=pl.lit(score_name),
            score_value=score_funs[score_name](pl.col("data"), pl.col("pred")),
        ).filter(pl.col("score_value").is_not_null())
        all_scores = pl.concat([all_scores, score])

    return all_scores


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
